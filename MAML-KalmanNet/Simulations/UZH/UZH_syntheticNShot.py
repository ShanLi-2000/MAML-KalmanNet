import math
import torch
import random
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

class SyntheticNShot:

    def __init__(self, batchsz, n_way, k_shot, k_shot_test, q_query, data_path, Is_GenData=False, use_cuda=False):
        # 定义模型
        self.x_dim = 9
        self.y_dim = 3
        self.data_path = data_path
        if use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.is_linear = True

        self.dt = 0.01
        temp_F = torch.tensor([[1., self.dt, 1 / 2 * self.dt ** 2],
                               [0., 1., self.dt],
                               [0., 0., 1.]])
        temp_H = torch.tensor([[0., 0., 1.]])
        self.F = torch.kron(torch.eye(3), temp_F).to(self.device)
        self.H = torch.kron(torch.eye(self.y_dim), temp_H).to(self.device)

        self.init_state_gen = torch.tensor([6.97259757e+00, 0.00000000e+00,  0.00000000e+00, 3.38774969e+00,
                                            0.00000000e+00, 0.00000000e+00, -8.47227610e-01, 0.00000000e+00,
                                            0.00000000e+00]).reshape(-1, 1)
        self.init_state_filter = self.init_state_gen.clone()
        self.noise_list_test = [2e-5, 6e-5, 2e-4, 6e-4, 2e-3, 6e-3, 2e-2, 6e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1]

        self.ekf_cov = torch.zeros((self.x_dim, self.x_dim)).reshape((1, self.x_dim, self.x_dim)).repeat(q_query, 1, 1).to(self.device)

        if Is_GenData:
            print("----Generating Data----")
            self.generate_data(seq_num=100, seq_len=100, mode='train')
            self.generate_data(seq_num=100, seq_len=200, mode='test')

        x_train_state = torch.load(self.data_path + 'train/state.pt')
        x_train_obs = torch.load(self.data_path + 'train/obs.pt')  # [list_length ** 2, seq_num, y_dim, seq_len]
        self.x_train = {"state": x_train_state, "obs": x_train_obs}

        x_test_state = torch.load(self.data_path + 'test/state.pt')
        x_test_obs = torch.load(self.data_path + 'test/obs.pt')
        self.x_test = {"state": x_test_state, "obs": x_test_obs}

        self.n_cls = x_train_state.shape[0]
        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_shot_test = k_shot_test
        self.q_query = q_query

        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}
        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"], mode='train'),
                               "test": self.load_data_cache(self.datasets["test"], mode='test')}

    def generate_data(self, seq_num, seq_len, mode='train'):
        if mode not in ['train', 'test']:
            raise ValueError('Possible mode = ["train", "test"]')
        if mode == 'train':
            noise_list = [5e-5, 7e-5, 1e-4, 5e-4, 7e-4, 1e-3, 5e-3, 7e-3, 1e-2, 5e-2, 7e-2, 1e-1, 5e-1, 7e-1, 1.]
        else:
            noise_list = self.noise_list_test
        state = torch.zeros((len(noise_list) ** 2, seq_num, self.x_dim, seq_len))
        obs = torch.zeros((len(noise_list) ** 2, seq_num, self.y_dim, seq_len))
        j = 0
        for q2 in noise_list:
            for r2 in noise_list:
                print("Generating Data: q2:" + str(q2) + "r2:" + str(r2))
                cov_q = q2 * torch.tensor([[1/4.*self.dt**4, 1/2.*self.dt**3, 1/2.*self.dt**2],
                                           [1/2.*self.dt**3,      self.dt**2,         self.dt],
                                           [1/2.*self.dt**2,      self.dt,                 1]])
                cov_q = torch.kron(torch.eye(3), cov_q)
                cov_q = cov_q + torch.eye(self.x_dim) * 1e-8  # ensure the matrix is positive define

                cov_r = r2 * torch.eye(self.y_dim)

                state_mtx = torch.zeros((seq_num, self.x_dim, seq_len))
                obs_mtx = torch.zeros((seq_num, self.y_dim, seq_len))
                x_prev = self.init_state_gen.unsqueeze(0)
                x_prev = x_prev.repeat(seq_num, 1, 1)
                with torch.no_grad():
                    for i in range(seq_len):
                        xt = self.f(x_prev)
                        x_mean = torch.zeros(seq_num, self.x_dim)
                        distrib = MultivariateNormal(loc=x_mean, covariance_matrix=cov_q)
                        eq = distrib.rsample().view(seq_num, self.x_dim, 1).to(self.device)
                        xt = torch.add(xt, eq)
                        yt = self.g(xt)
                        y_mean = torch.zeros(seq_num, self.y_dim)
                        distrib = MultivariateNormal(loc=y_mean, covariance_matrix=cov_r)
                        er = distrib.rsample().view(seq_num, self.y_dim, 1).to(self.device)
                        yt = torch.add(yt, er)
                        x_prev = xt.clone()
                        state_mtx[:, :, i] = torch.squeeze(xt, 2)
                        obs_mtx[:, :, i] = torch.squeeze(yt, 2)
                    state[j] = state_mtx
                    obs[j] = obs_mtx
                    j += 1

        torch.save(state, self.data_path + mode + '/state.pt')
        torch.save(obs, self.data_path + mode + '/obs.pt')

    def load_data_cache(self, data_dic_pack, mode='train'):

        state = data_dic_pack["state"]
        obs = data_dic_pack["obs"]
        noise_num = state.shape[0]
        if mode == 'train':
            k_shot = self.k_shot
        else:
            k_shot = self.k_shot_test
        sample_num = 10

        data_cache = []
        for sample in range(sample_num):
            state_spt, obs_spt, state_qry, obs_qry = [], [], [], []
            selected_class = np.random.choice(noise_num, self.n_way, False)

            for _, cur_class in enumerate(selected_class):
                selected_task_num = np.random.choice(state.shape[1], k_shot + self.q_query, False)

                state_spt.append(state[cur_class][selected_task_num[:k_shot]])
                obs_spt.append(obs[cur_class][selected_task_num[:k_shot]])
                state_qry.append(state[cur_class][selected_task_num[k_shot:]])
                obs_qry.append(obs[cur_class][selected_task_num[k_shot:]])

            perm = np.random.permutation(self.n_way)
            state_spt = np.array([t.detach().cpu().numpy() for t in state_spt])[perm]
            obs_spt = np.array([t.detach().cpu().numpy() for t in obs_spt])[perm]
            state_qry = np.array([t.detach().cpu().numpy() for t in state_qry])[perm]
            obs_qry = np.array([t.detach().cpu().numpy() for t in obs_qry])[perm]
            selected_class = selected_class[perm]

            data_cache.append([state_spt, obs_spt, state_qry, obs_qry, selected_class])

        return data_cache

    def next(self, mode="train"):
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch

    def f(self, x):
        F = self.F.reshape((1, self.x_dim, self.x_dim)).repeat(x.shape[0], 1, 1).to(x.device)
        return torch.bmm(F, x)

    def f_single(self, x):
        F = self.F.to(x.device)
        return F @ x

    def g(self, x):
        H = self.H.reshape((1, self.y_dim, self.x_dim)).repeat(x.shape[0], 1, 1).to(x.device)
        return torch.bmm(H, x)

    def g_single(self, x):
        H = self.H.to(x.device)
        return H @ x

    def Jacobian_f(self, x, is_seq=True):
        if is_seq:
            F = self.F.reshape((1, self.x_dim, self.x_dim)).repeat(x.shape[0], 1, 1)
        else:
            F = self.F
        return F.to(x.device)

    def Jacobian_g(self, x, is_seq=True):
        if is_seq:
            H = self.H.reshape((1, self.y_dim, self.x_dim)).repeat(x.shape[0], 1, 1)
        else:
            H = self.H
        return H.to(x.device)
