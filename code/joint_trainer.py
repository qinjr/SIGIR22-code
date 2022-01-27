import os
import pickle as pkl
from logging import getLogger
from time import time

import numpy as np
import torch
from torch.nn.modules import loss
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from recbole.utils import set_color

from utils.evaluations import TopKMetric, PointMetric
from utils.early_stopper import EarlyStopper
from utils.log import dict2str, get_tensorboard, get_local_time, ensure_dir
from loss import BPRLoss, RegLoss, TopKLoss, ICCLoss, BPR_LogLoss
from dataloader import Dataloader

class AbsJointTrainer(object):
    r"""Joint Trainer Class is used to manage the training and evaluation processes of recommender system models in independent joint mode.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, recsys, dataset_name):
        self.config = config
        self.recsys = recsys
        self.dataset_name = dataset_name

    def domain_adaptation(self):
        r"""domain adaptation of the succeeding stages.
        """
        raise NotImplementedError('Method [next] should be implemented.')

    def joint_training(self):
        r"""studen learning of the preceding stages.
        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self):
        r"""Evaluate the model based on the eval data.
        """
        raise NotImplementedError('Method [next] should be implemented.')

class jointTrainer(AbsJointTrainer):
    r"""joint trainer for the joint training phase
    """
    def __init__(self, config, recsys, dataset_name):
        super().__init__(config, recsys, dataset_name)
        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger, 'joint')
        self.rec_sizes = config['rec_sizes']
        self.score_sizes = config['score_sizes']
        self.stage_names = config['stage_names']
        self.training_stages_teacher = config['training_stages_teacher']
        self.training_stages_student = config['training_stages_student']
        self.topk_dict = config['topk_dict']
        self.eval_modes = config['eval_modes']
        self.score_sizes = config['score_sizes']
        

        self.alphas = config['alphas']
        self.learner = config['learner']
        self.learning_rates_student = config['learning_rates_student']
        self.learning_rates_teacher = config['learning_rates_teacher']
        self.use_logit = config['use_logit']
        self.l2_norm = config['l2_norm']
        self.weight_decay = config['weight_decay']
        # self.max_epochs = config['max_epochs']
        self.pair_epochs = config['pair_epochs'] # [!!!!!Just for simplicity]
        # how much training epoch will we conduct one evaluation
        self.clip_grad_norm = config['clip_grad_norm']
        self.batch_sizes = config['batch_sizes']
        self.teacher_batch_size = config['teacher_batch_size']

        self.device = torch.device(config['device'])

        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        self.saved_model_files = {}
        for i, model_name in enumerate(self.recsys.get_model_names()):
            file_name = '{}-{}-{}.pth'.format(model_name.lower(), dataset_name, get_local_time())
            self.saved_model_files[self.stage_names[i]] = os.path.join(self.checkpoint_dir, file_name)
        
        self.start_epoch = 0
        self.cur_step = 0
        self.continue_metrics = config['continue_metrics']
        self.student_epochs = config['student_epochs']
        self.teacher_epochs = config['teacher_epochs']
        self.total_rounds = config['total_rounds']
        self.teacher_way = config['teacher_way']
        self.teacher_loss_type = config['teacher_loss_type']
        self.teacher_rand_size = config['teacher_rand_size']
        self.teacher_first = config['teacher_first']

        self.teacher_optimizers = {}
        self.student_optimizers = {}
        for i, stage_name in enumerate(self.stage_names):
            optimizer = self._build_optimizer(self.recsys.get_model(i).parameters(), self.learning_rates_student[stage_name], 'student')
            self.student_optimizers[stage_name] = optimizer

            optimizer = self._build_optimizer(self.recsys.get_model(i).parameters(), self.learning_rates_teacher[stage_name], 'teacher')
            self.teacher_optimizers[stage_name] = optimizer

        # user hist
        if self.config['have_hist']:
            with open(self.config['hist_dict'] + 'train.pkl', 'rb') as f:
                self.hist_dict_train = pkl.load(f)
            with open(self.config['hist_dict'] + 'test.pkl', 'rb') as f:
                self.hist_dict_test = pkl.load(f)

            with open(self.config['hist_len_dict'] + 'train.pkl', 'rb') as f:
                self.hist_len_dict_train = pkl.load(f)
            with open(self.config['hist_len_dict'] + 'test.pkl', 'rb') as f:
                self.hist_len_dict_test = pkl.load(f)
        
        with open(self.config['teacher_pos_data'], 'rb') as f:
            self.teacher_pos_data = pkl.load(f)
            # self.teacher_pos_data = [torch.from_numpy(t) for t in self.teacher_pos_data]
    
    def _build_optimizer(self, params, lr, role = 'student'):
        r"""Init the Optimizer
        Returns:
            torch.optim: the optimizer
        """
        if self.learner[role].lower() == 'adam':
            optimizer = optim.Adam(params, lr=lr, weight_decay=self.weight_decay)
        elif self.learner[role].lower() == 'sgd':
            optimizer = optim.SGD(params, lr=lr, weight_decay=self.weight_decay)
        elif self.learner[role].lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=lr, weight_decay=self.weight_decay)
        elif self.learner[role].lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=lr, weight_decay=self.weight_decay)
        elif self.learner[role].lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=lr)
            if self.weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=lr)
        return optimizer
    
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')
    
    def _logging_recsys_eval(self, recsys_evals):
        for stage in self.stage_names:
            eval_output = set_color('eval result of stage-{}'.format(stage), 'blue') + ': \n' + dict2str(recsys_evals[stage])
            self.logger.info(eval_output)

    def _get_loss_func(self, loss_type):
        if loss_type == 'mse':
            return torch.nn.MSELoss()
        elif loss_type == 'topk':
            return TopKLoss()
        elif loss_type == 'll':
            return torch.nn.BCELoss()
        elif loss_type == 'bpr':
            return BPRLoss()
        elif loss_type == 'bpr_ll':
            return BPR_LogLoss()
        # elif self.loss_type == 'reg':
        #     return RegLoss()
    
    def _get_user_hist(self, x_user, stage = 'train'):
        uids = x_user[:,0].tolist()
        user_history = []
        hist_len = []
        if stage == 'train':
            for uid in uids:
                user_history.append(self.hist_dict_train[uid])
                hist_len.append(self.hist_len_dict_train[uid])
        elif stage == 'test':
            for uid in uids:
                user_history.append(self.hist_dict_test[uid])
                hist_len.append(self.hist_len_dict_test[uid])
        return torch.tensor(user_history), torch.tensor(hist_len)

    def _get_teacher_pair_data(self, teacher_stage_data, teacher_stage_name):
        def first_nonzero(arr, axis, invalid_val=-1):
            mask = arr!=0
            return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
        # generate negative dict
        users_tile_r, candidate_items_r, labels_r = teacher_stage_data
        list_len = self.score_sizes[teacher_stage_name]
        users_tile = users_tile_r.reshape([-1, list_len, users_tile_r.shape[1]])
        candidate_items = candidate_items_r.reshape([-1, list_len, candidate_items_r.shape[1]])
        labels = labels_r.reshape([-1, list_len])
        # print('prepared')
        
        # print('random sampling begin')
        # random sample firstly
        all_index = np.zeros_like(labels) + np.arange(labels.shape[1])
        random_select_index = np.random.rand(*all_index.shape).argsort(1)[:,:self.teacher_rand_size[teacher_stage_name]]
        random_select_index = np.sort(random_select_index, axis=1)
        labels = np.take_along_axis(labels, random_select_index, axis=1)
        random_select_index = np.tile(np.expand_dims(random_select_index,axis=2),(1,1,candidate_items.shape[2]))
        candidate_items = np.take_along_axis(candidate_items, random_select_index, axis=1)

        # select the first negative sample
        uids = users_tile[:, 0, 0]
        select_index = first_nonzero(1 - labels, axis=1).reshape(-1, 1) #np.zeros((labels.shape[0], 1))#
        select_index = np.tile(np.expand_dims(select_index,axis=2),(1,1,candidate_items.shape[2]))

        neg_dict = {}
        selected_neg_item = np.squeeze(np.take_along_axis(candidate_items, select_index, axis=1))
        for i, uid in enumerate(uids):
            neg_dict[uid] = selected_neg_item[i].reshape(1, -1)

        # generate pair-wise data for teacher
        x_user, x_item = self.teacher_pos_data
        x_user_uid = x_user[:,0].tolist()
        x_item_neg = []
        
        for uid in x_user_uid:
            item_neg = neg_dict[uid]
            x_item_neg.append(item_neg.flatten().tolist()) 
        x_item_neg = np.array(x_item_neg)
        return [x_user, x_item, x_item_neg]


    def _get_teacher_pair_data2(self, teacher_stage_data, teacher_stage_name):
        def first_nonzero(arr, axis, invalid_val=-1):
            mask = arr!=0
            return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
        # generate negative dict
        users_tile_r, candidate_items_r, labels_r = teacher_stage_data
        list_len = self.score_sizes[teacher_stage_name]
        users_tile = users_tile_r.reshape([-1, list_len, users_tile_r.shape[1]])
        candidate_items = candidate_items_r.reshape([-1, list_len, candidate_items_r.shape[1]])
        labels = labels_r.reshape([-1, list_len])
        # print('prepared')
        
        users_select = np.concatenate((users_tile[:,:2,:][:,0,:], users_tile[:,:2,:][:,1,:]), axis=0)
        items_select = np.concatenate((candidate_items[:,:2,:][:,0,:], candidate_items[:,:2,:][:,1,:]), axis=0)
        labels_select = np.concatenate((labels[:,:2][:,0], labels[:,:2][:,1]), axis=0)

        print(users_select.shape)
        print(items_select.shape)
        print(labels_select.shape)

        # generate pair-wise data for teacher
        x_user_pos, x_item_pos = self.teacher_pos_data
        x_user = np.concatenate([users_select, x_user_pos], axis=0)
        x_item = np.concatenate([items_select, x_item_pos], axis=0)
        labels = np.concatenate([labels_select, np.ones(labels_select, shape=(x_user_pos.shape[0],))], axis=0)
        print(x_user.shape)
        print(x_item.shape)
        print(labels.shape)
        
        return [x_user, x_item, labels]


    def _get_teacher_pos_data(self):
        x_user, x_item = self.teacher_pos_data
        labels = np.ones([x_user.shape[0],], np.float32)
        return x_user, x_item, labels
    
    def _get_teacher_neg_data(self, data_tuple):
        x_user, x_item, label = data_tuple
        mask = (label == 0)
        return x_user[mask], x_item[mask], label[mask]

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss):
        des = 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        
        des = '%.' + str(des) + 'f'
        train_loss_output += set_color('train loss', 'blue') + ': ' + des % loss
        return train_loss_output + ']'

    def _save_checkpoint(self, model, stage_name):
        state = {
            'state_dict': model.state_dict(),
        }
        torch.save(state, self.saved_model_files[stage_name])

    def _student_learning_epoch(self, student_stage, teacher_stage, dataloader, \
                                    teacher, student, epoch_idx, \
                                    show_progress=True):
        teacher.eval()
        student.train()

        iter_data = (
            tqdm(
                dataloader,
                total=len(dataloader),
                ncols=100,
                desc=set_color(f"Training: {epoch_idx:>5}", 'pink'),
                position=0, 
                leave=True
            ) if show_progress else dataloader
        )

        mse_loss_func = self._get_loss_func('mse')
        ll_loss_func = self._get_loss_func('ll')
        topk_loss_func = self._get_loss_func('topk')

        total_loss = None
        optimizer = self.student_optimizers[student_stage]
        k = self.rec_sizes[teacher_stage]
        list_len = self.score_sizes[student_stage]
        
        for batch_idx, batch_data in enumerate(iter_data):
            optimizer.zero_grad()
            x_user, x_item, label = batch_data
            user_hist = None
            hist_len = None
            if teacher.use_hist or student.use_hist:
                user_hist, hist_len = self._get_user_hist(x_user)
                user_hist = user_hist.to(self.device)
                hist_len = hist_len.to(self.device)

            x_user = torch.from_numpy(x_user).to(self.device)
            x_item = torch.from_numpy(x_item).to(self.device)
            label = torch.from_numpy(label).to(self.device)

            # get teacher output as target
            with torch.no_grad():
                pred_t = teacher(x_user, x_item, user_hist, hist_len)
                
            # train student
            pred_s = student(x_user, x_item, user_hist, hist_len)
            
            pred_s_logit = torch.logit(pred_s, eps = 1e-6)
            pred_t_logit = torch.logit(pred_t, eps = 1e-6)
            
            if self.use_logit:
                loss = self.alphas[student_stage] * topk_loss_func(pred_s_logit, pred_t_logit, k, list_len) + (1 - self.alphas[student_stage]) * mse_loss_func(pred_s_logit, pred_t_logit)
            else:
                loss = self.alphas[student_stage] * topk_loss_func(pred_s, pred_t, k, list_len) + (1 - self.alphas[student_stage]) * mse_loss_func(pred_s, pred_t)
            # loss =  ll_loss_func(pred_s, pred_t)
            # loss = mse_loss_func(pred_s, pred_t)

            total_loss = loss.item() if total_loss is None else total_loss + loss.item()
            self._check_nan(loss)

            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(student.parameters(), **self.clip_grad_norm)
            optimizer.step()
        return total_loss / len(dataloader)

    def _teacher_learning_epoch(self, teacher_stage, dataloader, \
                                    teacher, epoch_idx, show_progress=True):
        teacher.train()
        iter_data = (
            tqdm(
                dataloader,
                total=len(dataloader),
                ncols=100,
                desc=set_color(f"Training: {epoch_idx:>5}", 'pink'),
                position=0, 
                leave=True
            ) if show_progress else dataloader
        )

        ll_loss_func = self._get_loss_func('ll')
        
        total_loss = None
        optimizer = self.teacher_optimizers[teacher_stage]
        
        for batch_idx, batch_data in enumerate(iter_data):
            optimizer.zero_grad()
            
            x_user, x_item, label = batch_data
            user_hist = None
            hist_len = None
            if teacher.use_hist:
                user_hist, hist_len = self._get_user_hist(x_user)
                user_hist = user_hist.to(self.device)
                hist_len = hist_len.to(self.device)

            x_user = torch.from_numpy(x_user).to(self.device)
            x_item = torch.from_numpy(x_item).to(self.device)
            label = torch.from_numpy(label).to(self.device)

            pred_t = teacher(x_user, x_item, user_hist, hist_len)
            
            loss = ll_loss_func(pred_t, label)
            total_loss = loss.item() if total_loss is None else total_loss + loss.item()
            self._check_nan(loss)

            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(teacher.parameters(), **self.clip_grad_norm)
            optimizer.step()
        return total_loss / len(dataloader)

    def _teacher_learning_pair_epoch(self, teacher_stage, stage_data, \
                                    teacher, epoch_idx, show_progress=True):
        # prepare the teacher data
        data_tuple = self._get_teacher_pair_data(stage_data, teacher_stage)
        # data_tuple = self._get_teacher_pair_data2(stage_data, teacher_stage)
        dataloader = Dataloader(data_tuple, self.teacher_batch_size, True)

        teacher.train()
        iter_data = (
            tqdm(
                dataloader,
                total=len(dataloader),
                ncols=100,
                desc=set_color(f"Training: {epoch_idx:>5}", 'pink'),
                position=0, 
                leave=True
            ) if show_progress else dataloader
        )

        loss_type = self.teacher_loss_type[teacher_stage]
        loss_func = self._get_loss_func(loss_type)
        
        total_loss = None
        optimizer = self.teacher_optimizers[teacher_stage]
        
        for batch_idx, batch_data in enumerate(iter_data):
            optimizer.zero_grad()
            
            x_user, x_item, x_item_neg = batch_data
            # x_user, x_item, labels = batch_data
            user_hist = None
            hist_len = None
            if teacher.use_hist:
                user_hist, hist_len = self._get_user_hist(x_user)
                user_hist = user_hist.to(self.device)
                hist_len = hist_len.to(self.device)

            x_user = torch.from_numpy(x_user).to(self.device)
            x_item = torch.from_numpy(x_item).to(self.device)
            x_item_neg = torch.from_numpy(x_item_neg).to(self.device)
            # labels = torch.from_numpy(labels).to(self.device)

            pred_t_pos = teacher(x_user, x_item, user_hist, hist_len)
            pred_t_neg = teacher(x_user, x_item_neg, user_hist, hist_len)
            # pred_t = teacher(x_user, x_item, user_hist, hist_len)
            
            loss = loss_func(pred_t_pos, pred_t_neg)
            # loss = loss_func(pred_t, labels)
            total_loss = loss.item() if total_loss is None else total_loss + loss.item()
            self._check_nan(loss)

            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(teacher.parameters(), **self.clip_grad_norm)
            optimizer.step()
        return total_loss / len(dataloader)


    def _student_learning(self, stage_data, round_id):
        self.logger.info('\n\nstudent learning of round {} begins'.format(round_id))
        for i, stage_name in enumerate(self.stage_names[:-1]): # all the possible student stages
            if stage_name not in self.training_stages_student:
                continue
            self.logger.info('student:{} training begin with {} as teacher...'.format(stage_name, self.stage_names[i + 1]))
            teacher = self.recsys.get_model(i + 1)
            student = self.recsys.get_model(i)
            data_s = stage_data[i]
            dl_s = Dataloader(data_s, self.batch_sizes[stage_name], shuffle=False)
            
            for e in range(self.student_epochs[stage_name]):
                training_start_time = time()
                teacher_stage_name = self.stage_names[i + 1]
                student_train_loss = self._student_learning_epoch(stage_name, teacher_stage_name, \
                                    dl_s, teacher, student, e, \
                                    show_progress=True)
                
                training_end_time = time()
                train_loss_output = \
                    self._generate_train_loss_output(e, training_start_time, training_end_time, student_train_loss)
                self.logger.info(train_loss_output)
                dl_s.refresh()


            self.recsys.update_model(i, student)
            if i == 0:
                self.logger.info('faiss index is being updated...')
                self.recsys.update_faiss_index()
            
            self.logger.info('RecSys testing after student {} training round {}'.format(stage_name, round_id))
            recsys_evals = self.recsys.test_all_stages('test')
            self._logging_recsys_eval(recsys_evals)
        return recsys_evals

    def _teacher_learning(self, stage_data, round_id):
        self.logger.info('\n\nteacher learning of round {} begins'.format(round_id))
        for i, stage_name in enumerate(self.stage_names): # all the possible teacher stages
            if stage_name not in self.training_stages_teacher:
                continue

            self.logger.info('teacher:{} training begin...'.format(stage_name))
            teacher = self.recsys.get_model(i)
            data_t = stage_data[i]
            dl_t = Dataloader(data_t, self.teacher_batch_size, shuffle=False)
            x_user_pos, x_item_pos, label_pos = self._get_teacher_pos_data()
            x_user_neg, x_item_neg, label_neg = self._get_teacher_neg_data(data_t)
            data_t_all = [np.concatenate((x_user_pos, x_user_neg), axis=0), np.concatenate((x_item_pos, x_item_neg), axis=0), np.concatenate((label_pos, label_neg), axis=0)]
            dl_t_all = Dataloader(data_t_all, self.teacher_batch_size, shuffle=True)

            for e in range(self.teacher_epochs[stage_name]):
                training_start_time = time()
                if self.teacher_way[stage_name] == 'normal':
                    teacher_train_loss = self._teacher_learning_epoch(stage_name, dl_t, \
                                        teacher, e, show_progress=True)
                    dl_t.refresh()
                elif self.teacher_way[stage_name] == 'all':
                    teacher_train_loss = self._teacher_learning_epoch(stage_name, dl_t_all, \
                                        teacher, e, show_progress=True)
                    dl_t_all.refresh()
                elif self.teacher_way[stage_name] == 'gen_pair':
                    teacher_train_loss = self._teacher_learning_pair_epoch(stage_name, data_t, teacher, e, 
                                            show_progress=True)
                training_end_time = time()
                train_loss_output = \
                    self._generate_train_loss_output(e, training_start_time, training_end_time, teacher_train_loss)
                self.logger.info(train_loss_output)
                
            self.recsys.update_model(i, teacher)
        
            self.logger.info('RecSys testing after teacher {} training round {}'.format(stage_name, round_id))
            recsys_evals = self.recsys.test_all_stages('test')
            self._logging_recsys_eval(recsys_evals)
        return recsys_evals

    def joint_training2(self):
        self.logger.info('initial testing')
        recsys_evals = self.recsys.test_all_stages('test')
        for stage in self.stage_names:
            eval_output = set_color('eval result of stage-{}'.format(stage), 'blue') + ': \n' + dict2str(recsys_evals[stage])
            self.logger.info(eval_output)

        self.logger.info('joint training begins')
        for round_id in range(self.total_rounds):
            if self.teacher_first:
                # teacher learning
                stage_data = self.recsys.gen_stage_data('train')
                recsys_evals = self._teacher_learning(stage_data, round_id)
                # student learning
                stage_data = self.recsys.gen_stage_data('train')
                recsys_evals = self._student_learning(stage_data, round_id)
            else:
                # student learning
                stage_data = self.recsys.gen_stage_data('train')
                recsys_evals = self._student_learning(stage_data, round_id)
                # teacher learning
                stage_data = self.recsys.gen_stage_data('train')
                recsys_evals = self._teacher_learning(stage_data, round_id)

        stage_data = self.recsys.gen_stage_data('train')
        recsys_evals = self._teacher_learning(stage_data, round_id)


    def joint_training(self, verbose=True, saved=True, show_progress=True):
        recsys_evals_list = []
        # early_stopper = EarlyStopper(self.config)

        # start with testing
        self.logger.info('Initial testing')
        recsys_evals = self.recsys.test_all_stages('test')
        # recsys_evals_list.append(recsys_evals)
        if verbose:
            for stage in self.stage_names:
                eval_output = set_color('eval result of stage-{}'.format(stage), 'blue') + ': \n' + dict2str(recsys_evals[stage])
                self.logger.info(eval_output)

        self.logger.info('student learning begins')
        
        for i, stage_name in enumerate(self.stage_names[:-1]):
            if stage_name not in self.training_stages_student:
                continue
            self.logger.info('\n\nlearning of stage pair:{} and {} begin...'.format(stage_name, self.stage_names[i + 1]))
            teacher = self.recsys.get_model(i + 1)
            student = self.recsys.get_model(i)

            for epoch_idx in range(self.start_epoch, self.pair_epochs[stage_name][0]):
                self.logger.info('pair training epoch: {}'.format(epoch_idx))

                '''teacher learning'''
                stage_data = self.recsys.gen_stage_data('train')
                for e in range(self.pair_epochs[stage_name][1]):
                    data_t = stage_data[i + 1]
                    training_start_time = time()
                    teacher_stage_name = self.stage_names[i + 1]
                    if self.teacher_way[teacher_stage_name] == 'normal':
                        dl_t = Dataloader(data_t, self.teacher_batch_size, shuffle=False)
                        teacher_train_loss = self._teacher_learning_epoch(teacher_stage_name, dl_t, \
                                            teacher, e, show_progress=True)
                        dl_t.refresh()
                    elif self.teacher_way[teacher_stage_name] == 'gen_pair':
                        teacher_train_loss = self._teacher_learning_pair_epoch(teacher_stage_name, data_t, teacher, e, 
                                                show_progress=True)
                    
                    training_end_time = time()
                    train_loss_output = \
                        self._generate_train_loss_output(e, training_start_time, training_end_time, teacher_train_loss)
                    if verbose:
                        self.logger.info(train_loss_output)
                    self.tensorboard.add_scalar('Loss/Joint-Train-{}'.format(teacher.get_name()), teacher_train_loss, epoch_idx)
                self.recsys.update_model(i + 1, teacher)

                '''student learning'''
                stage_data = self.recsys.gen_stage_data('train')
                for e in range(self.pair_epochs[stage_name][2]):
                    data_s = stage_data[i]
                    dl_s = Dataloader(data_s, self.batch_sizes[stage_name], shuffle=False)
                    
                    training_start_time = time()
                    teacher_stage_name = self.stage_names[i + 1]
                    student_train_loss = self._student_learning_epoch(stage_name, teacher_stage_name, \
                                        dl_s, teacher, student, e, \
                                        show_progress=show_progress)
                    
                    training_end_time = time()
                    train_loss_output = \
                        self._generate_train_loss_output(e, training_start_time, training_end_time, student_train_loss)
                    if verbose:
                        self.logger.info(train_loss_output)
                    self.tensorboard.add_scalar('Loss/Joint-Train-{}'.format(student.get_name()), student_train_loss, epoch_idx)
                # update recsys model and if recall is trained, update faiss index
                self.recsys.update_model(i, student)
                if i == 0:
                    self.logger.info('faiss index is being updated...')
                    self.recsys.update_faiss_index()
                

                # eval
                eval_start_time = time()
                recsys_evals = self.recsys.test_all_stages('test')
                eval_end_time = time()
                recsys_evals_list.append(recsys_evals)

                continue_metric_value = recsys_evals[stage_name][self.continue_metrics[stage_name]]
                continue_metric_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color(self.continue_metrics[stage_name], 'blue') + ": %f]") % \
                                     (epoch_idx, eval_end_time - eval_start_time, continue_metric_value)
                self.logger.info(continue_metric_output)
                self.tensorboard.add_scalar('eval_{}_{}'.format(stage_name, self.continue_metrics[stage_name]), continue_metric_value, epoch_idx)

                if verbose:
                    for stage in self.stage_names:
                        eval_output = set_color('eval result of stage-{}'.format(stage), 'blue') + ': \n' + dict2str(recsys_evals[stage])
                        self.logger.info(eval_output)
                
                # continue_flag, save_flag = early_stopper.check(continue_metric_value, epoch_idx)
                
                # if save_flag and saved:
                #     self._save_checkpoint(student, stage_name)
                #     save_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_files[stage_name]
                #     if verbose:
                #         self.logger.info(save_output)

                # if not continue_flag:
                #     break
                self._save_checkpoint(student, stage_name)

        # final test
        self.logger.info('begin final testing...')
        recsys_evals = self.recsys.test_all_stages('test')
        recsys_evals_list.append(recsys_evals)
        if verbose:
            for stage in self.stage_names:
                eval_output = set_color('eval result of stage-{}'.format(stage), 'blue') + ': \n' + dict2str(recsys_evals[stage])
                self.logger.info(eval_output)


class ICCTrainer(AbsJointTrainer):
    r"""Independent trainer for the warmup training phase
    """
    def __init__(self, config, recsys, dataset_name):
        super().__init__(config, recsys, dataset_name)
        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        # self.train_mode = config['train_mode']
        # self.batch_neg_size = config['batch_neg_size']
        self.loss_type = config['loss_type']
        self.learner = config['icc_learner']
        self.learning_rate = config['learning_rate_icc']
        self.l2_norm = config['l2_norm']
        self.max_epochs = config['max_epochs']
        # how much training epoch will we conduct one evaluation
        self.eval_step = min(config['eval_step'], self.max_epochs)
        self.clip_grad_norm = config['clip_grad_norm']
        self.stage_names = config['stage_names']

        self.eval_batch_size = config['eval_batch_size']
        self.eval_modes = config['eval_modes']
        self.device = torch.device(config['device'])
        self.checkpoint_dir = config['checkpoint_dir']
        # ensure_dir(self.checkpoint_dir)
        # saved_model_file = '{}-{}-{}.pth'.format(self.model.get_name().lower(), dataset_name, get_local_time())
        # self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config['weight_decay']
        self.continue_metrics = config['continue_metrics']

        self.start_epoch = 0
        self.cur_step = 0
        self.continue_metrics = config['continue_metrics']
        self.best_eval_result = None
        # self.train_loss_dict = dict()

        self.opt_params = []
        for i in range(len(self.stage_names)):
            params = list(self.recsys.get_model(i).parameters())
            self.opt_params += params
        
        self.optimizer = self._build_optimizer(self.opt_params)
        

        # user hist
        if self.config['have_hist']:
            with open(self.config['hist_dict'] + 'train.pkl', 'rb') as f:
                self.hist_dict_train = pkl.load(f)
            with open(self.config['hist_dict'] + 'test.pkl', 'rb') as f:
                self.hist_dict_test = pkl.load(f)

            with open(self.config['hist_len_dict'] + 'train.pkl', 'rb') as f:
                self.hist_len_dict_train = pkl.load(f)
            with open(self.config['hist_len_dict'] + 'test.pkl', 'rb') as f:
                self.hist_len_dict_test = pkl.load(f)


    def _build_optimizer(self, params):
        r"""Init the Optimizer
        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=self.learning_rate)
            if self.weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=self.learning_rate)
        return optimizer
    
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _get_loss_func(self, loss_type):
        if loss_type == 'll':
            return torch.nn.BCELoss()
        elif loss_type == 'bpr':
            return BPRLoss()
        elif loss_type == 'reg':
            return RegLoss()
        elif loss_type == 'icc':
            return ICCLoss()
    
    def _get_user_hist(self, x_user, stage = 'train'):
        uids = x_user[:,0].tolist()
        user_history = []
        hist_len = []
        if stage == 'train':
            for uid in uids:
                user_history.append(self.hist_dict_train[uid])
                hist_len.append(self.hist_len_dict_train[uid])
        elif stage == 'test':
            for uid in uids:
                user_history.append(self.hist_dict_test[uid])
                hist_len.append(self.hist_len_dict_test[uid])
        return torch.tensor(user_history), torch.tensor(hist_len)
    
    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss):
        des = 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        
        des = '%.' + str(des) + 'f'
        train_loss_output += set_color('train loss', 'blue') + ': ' + des % loss
        return train_loss_output + ']'

    def _train_epoch(self, train_dl, epoch_idx, show_progress=True):
        models = []
        use_hist = False
        for i in range(len(self.stage_names)):
            self.recsys.get_model(i).train()
            models.append(self.recsys.get_model(i))
            if models[i].use_hist:
                use_hist = True
        
        main_loss_func = self._get_loss_func('icc')
        reg_loss_func = self._get_loss_func('reg')
        total_loss = None
        iter_data = (
            tqdm(
                train_dl,
                total=len(train_dl),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
                position=0, 
                leave=True
            ) if show_progress else train_dl
        )

        for batch_idx, batch_data in enumerate(iter_data):
            self.optimizer.zero_grad()
            x_user, x_item, y = batch_data
            user_hist = None
            hist_len = None
            if use_hist:
                user_hist, hist_len = self._get_user_hist(x_user)
                user_hist = user_hist.to(self.device)
                hist_len = hist_len.to(self.device)
            x_user = x_user.to(self.device)
            x_item = x_item.to(self.device)
            y = y.float().to(self.device)

            pred_models = None
            for i in range(len(self.stage_names)):
                model = models[i]
                pred = model(x_user, x_item, user_hist, hist_len).unsqueeze(dim=1)
                pred_models = pred if pred_models == None else torch.cat((pred_models, pred), dim=1)
            
            loss = main_loss_func(pred_models, y) + self.l2_norm * reg_loss_func(self.opt_params)
        
            total_loss = loss.item() if total_loss is None else total_loss + loss.item()
            self._check_nan(loss)
            loss.backward()

            if self.clip_grad_norm:
                clip_grad_norm_(self.opt_params, **self.clip_grad_norm)
            self.optimizer.step()
        
        for i in range(len(self.stage_names)):
            self.recsys.update_model(i, models[i])
            if i == 0:
                self.recsys.update_faiss_index()

        return total_loss / len(train_dl)
    
    def fit(self, train_dl, verbose=True, saved=True, show_progress=True):
        early_stopper = EarlyStopper(self.config)
        
        recsys_evals_list = []
        self.logger.info('Initial testing')
        recsys_evals = self.recsys.test_all_stages('test')
        # recsys_evals_list.append(recsys_evals)
        if verbose:
            for stage in self.stage_names:
                eval_output = set_color('eval result of stage-{}'.format(stage), 'blue') + ': \n' + dict2str(recsys_evals[stage])
                self.logger.info(eval_output)
            
        
        for epoch_idx in range(self.start_epoch, self.max_epochs):
            if epoch_idx > self.start_epoch:
                train_dl.refresh()
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_dl, epoch_idx, show_progress=show_progress)
            # self.train_loss_dict[epoch_idx] = train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self.tensorboard.add_scalar('Loss/Train', train_loss, epoch_idx)

            # eval
            self.logger.info('Testing...')
            recsys_evals = self.recsys.test_all_stages('test')
            recsys_evals_list.append(recsys_evals)
            if verbose:
                for stage in self.stage_names:
                    eval_output = set_color('eval result of stage-{}'.format(stage), 'blue') + ': \n' + dict2str(recsys_evals[stage])
                    self.logger.info(eval_output)
            
            continue_metric_value = recsys_evals['ranking'][self.continue_metrics['ranking']]
            self.tensorboard.add_scalar('eval_{}'.format(self.continue_metrics['ranking']), continue_metric_value, epoch_idx)
            continue_flag, _ = early_stopper.check(continue_metric_value, epoch_idx)

            if not continue_flag:
                break
    
        best_epoch = early_stopper.get_best_epoch_idx()
        best_eval_result = recsys_evals_list[best_epoch]
        for stage in self.stage_names:
            eval_output = set_color('eval result of stage-{}'.format(stage), 'blue') + ': \n' + dict2str(best_eval_result[stage])
            self.logger.info(eval_output)
        return best_eval_result



