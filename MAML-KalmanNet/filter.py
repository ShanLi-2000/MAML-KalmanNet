import torch
import math
from torch import nn
from torch import optim
from state_dict_learner import Learner

class Filter:
    def __init__(self, args, model, is_linear_net=True):
        self.update_lr = args.update_lr

        self.model = model
        self.args = args
        if args.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise Exception("No GPU found, please set args.use_cuda = False")
        else:
            self.device = torch.device('cpu')

        self.train_net = Learner(self.model.x_dim, self.model.y_dim, args, is_linear_net).to(self.device)
        self.train_optim = optim.Adam(self.train_net.parameters(), lr=self.update_lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.train_optim,
                                                                        T_0=args.update_step, T_mult=1, eta_min=1e-6)
        self.loss_fn = nn.MSELoss()

        self.ekf_cov = self.model.ekf_cov
        self.ekf_cov_post = self.ekf_cov.detach().clone()
        self.ekf_state_history = torch.zeros((args.q_qry, self.model.x_dim, 1))

        self.state_history = torch.zeros((args.batch_size, self.model.x_dim, 1)).to(self.device)
        self.y_predict_history = torch.zeros((args.batch_size, self.model.y_dim, 1)).to(self.device)

        self.data_idx = 0
        self.batch_size = args.batch_size

    def compute_x_post(self, state, obs, task_net=None):  # obs:[seq_num, y_dim, seq_len]
        if task_net is None:
            temp_net = self.train_net
        else:
            temp_net = task_net
        self.reset_net()
        temp_net.initialize_hidden()
        seq_num, y_dim, seq_len = obs.shape

        if self.data_idx + self.batch_size >= seq_num:
            self.data_idx = 0
            shuffle_idx = torch.randperm(seq_num)
            state = state[shuffle_idx]
            obs = obs[shuffle_idx]
        batch_x = state[self.data_idx:self.data_idx+self.batch_size]
        batch_y = obs[self.data_idx:self.data_idx+self.batch_size]
        self.temp_batch_y = batch_y  # use for unsupervised KalmanNet

        # self.state_post = self.model.init_state_filter.reshape((1, self.model.x_dim, 1)).repeat(self.batch_size, 1, 1).to(state.device)
        self.state_post = batch_x[:, :, 0].reshape((batch_x.shape[0], batch_x.shape[1], 1))
        for i in range(1, seq_len):
            self.filtering(batch_y[:, :, i].unsqueeze(dim=2), temp_net)
        state_filtering = self.state_history
        self.temp_y_predict_history = self.y_predict_history  # use for unsupervised KalmanNet
        self.reset_net()
        temp_net.initialize_hidden()

        loss = self.loss_fn(state_filtering[:, :, 1:], batch_x[:, :, 1:])

        self.data_idx += self.batch_size

        return loss

    def compute_x_post_qry(self, state, obs, task_net=None):  # obs:[seq_num, y_dim, seq_len]
        if task_net is None:
            temp_net = self.train_net
        else:
            temp_net = task_net
        self.reset_net(is_train=False)
        temp_net.initialize_hidden(is_train=False)
        seq_num, y_dim, seq_len = obs.shape

        # self.state_post = self.model.init_state_filter.reshape((1, self.model.x_dim, 1)).repeat(seq_num, 1, 1).to(state.device)
        self.state_post = state[:, :, 0].reshape((state.shape[0], state.shape[1], 1))
        for i in range(1, seq_len):
            self.filtering(obs[:, :, i].unsqueeze(dim=2), temp_net)
        state_filtering = self.state_history
        self.reset_net(is_train=False)
        temp_net.initialize_hidden(is_train=False)

        loss = self.loss_fn(state_filtering[:, :, 1:], state[:, :, 1:])
        self.state_predict = state_filtering  # use for plot_trajectory
        return loss

    def filtering(self, observation, task_net):

        if self.training_first:
            self.state_post_past = self.state_post.detach().clone()

        x_last = self.state_post
        x_predict = self.model.f(x_last)

        if self.training_first:
            self.state_pred_past = x_predict.detach().clone()
            self.obs_past = observation.detach().clone()

        F_jacob = self.model.Jacobian_f(x_last)
        H_jacob = self.model.Jacobian_g(x_predict)

        y_predict = self.model.g(x_predict)
        residual = observation - y_predict

        # input 1: x_{k-1 | k-1} - x_{k-1 | k-2}
        state_inno = self.state_post_past - self.state_pred_past
        # input 2: residual
        # input 3: x_k - x_{k-1}
        diff_state = self.state_post - self.state_post_past
        # input 4: y_k - y_{k-1}
        diff_obs = observation - self.obs_past

        K_gain = task_net(diff_obs, residual, diff_state, state_inno, F_jacob, H_jacob)

        x_post = x_predict + torch.matmul(K_gain, residual)

        self.training_first = False
        self.state_pred_past = x_predict.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), 2)

        x_predict_k = self.model.f(x_post)
        y_predict_k = self.model.g(x_predict_k)
        self.y_predict_history = torch.cat((self.y_predict_history, y_predict_k.clone()), 2)

    def reset_net(self, is_train=True):
        self.training_first = True
        # self.train_net.initialize_hidden(is_train=is_train)
        if is_train:
            self.state_post = torch.zeros((self.batch_size, self.model.x_dim, 1)).to(self.device)
            self.y_predict_history = torch.zeros((self.batch_size, self.model.y_dim, 1)).to(self.device)
        else:
            self.state_post = torch.zeros((self.args.q_qry, self.model.x_dim, 1)).to(self.device)
            self.y_predict_history = torch.zeros((self.args.q_qry, self.model.y_dim, 1)).to(self.device)

        self.state_history = self.state_post.clone()

    def EKF(self, state, obs, cov_q, cov_r):
        seq_num, y_dim, seq_len = obs.shape
        with torch.no_grad():
            # self.ekf_state_post = self.model.init_state_filter.reshape((1, self.model.x_dim, 1)).repeat(seq_num, 1, 1)
            self.ekf_state_post = state[:, :, 0].reshape((state.shape[0], state.shape[1], 1))
            for i in range(1, seq_len):
                self.ekf_filtering(obs[:, :, i].unsqueeze(dim=2), cov_q, cov_r)
            state_filtering = self.ekf_state_history
            self.reset_ekf()

            loss = self.loss_fn(state_filtering[:, :, 1:], state[:, :, 1:])
            self.state_EKF = state_filtering  # use for EKF compute_trajectory

        return 10 * torch.log10(loss)

    def ekf_filtering(self, observation, Q, R):
        x_last = self.ekf_state_post
        x_predict = self.model.f(x_last)

        y_predict = self.model.g(x_predict)
        residual = observation - y_predict

        F_jacob = self.model.Jacobian_f(x_last)
        H_jacob = self.model.Jacobian_g(x_predict)
        cov_pred = (torch.bmm(torch.bmm(F_jacob, self.ekf_cov_post), torch.transpose(F_jacob, 1, 2))) \
                   + Q.reshape((1, Q.shape[0], Q.shape[0])).repeat(F_jacob.shape[0], 1, 1)
        temp = torch.linalg.inv(torch.bmm(torch.bmm(H_jacob, cov_pred), torch.transpose(H_jacob, 1, 2))
                                + R.reshape((1, R.shape[0], R.shape[0])).repeat(F_jacob.shape[0], 1, 1))
        K_gain = torch.bmm(torch.bmm(cov_pred, torch.transpose(H_jacob, 1, 2)), temp)

        x_post = x_predict + torch.bmm(K_gain, residual)

        cov_post = torch.bmm((torch.eye(self.model.x_dim) - torch.bmm(K_gain, H_jacob)), cov_pred)

        self.ekf_state_post = x_post.detach().clone()
        self.ekf_cov_post = cov_post.detach().clone()
        self.ekf_state_history = torch.cat((self.ekf_state_history, x_post.clone()), 2)

        self.ekf_pk = cov_pred

    def reset_ekf(self):
        self.ekf_state_post = torch.zeros((self.args.q_qry, self.model.x_dim, 1))
        self.ekf_cov_post = self.ekf_cov.detach().clone()
        self.ekf_state_history = self.ekf_state_post
