import math
import torch
import random
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

class SyntheticNShot:

    def __init__(self, batchsz, n_way, k_shot, k_shot_test, q_query, data_path, Is_GenData=False, use_cuda=False, Is_linear=True):
        # 定义模型
        if use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.data_path = data_path
        self.x_dim = 2
        self.y_dim = 2
        theta = 10 * 2 * math.pi / 360
        self.is_linear = Is_linear
        self.init_state_filter = torch.zeros(self.x_dim, 1)
        self.F = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]).to(self.device)
        self.H = torch.eye(self.y_dim, self.x_dim).to(self.device)
        self.noise_list_test = [0.0002, 0.002, 0.02, 0.2, 2., 4., 6., 8., 10., 12.]

        self.ekf_cov = torch.zeros((self.x_dim, self.y_dim)).reshape((1, self.x_dim, self.y_dim)).repeat(q_query, 1, 1)

        if Is_GenData:
            print("----Generating Data----")
            # generating training data
            self.generate_data(seq_num=100, seq_len=30, mode='train')
            # 生成测试数据
            self.generate_data(seq_num=100, seq_len=50, mode='test')

        x_train_state = torch.load(self.data_path + 'train/state.pt')
        x_train_obs = torch.load(self.data_path + 'train/obs.pt')  # [length_list ** 2, seq_num, y_dim, seq_len]
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
            noise_list = [0.0005, 0.0007, 0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1., 3., 5., 7., 10.]
        else:
            noise_list = self.noise_list_test
        state = torch.zeros((len(noise_list) ** 2, seq_num, self.x_dim, seq_len))
        obs = torch.zeros((len(noise_list) ** 2, seq_num, self.y_dim, seq_len))
        j = 0
        for q2 in noise_list:
            for r2 in noise_list:
                print("Generating Data: q2:" + str(q2) + "r2:" + str(r2))
                cov_q = q2 * torch.eye(self.x_dim)
                cov_r = r2 * torch.eye(self.y_dim)
                state_mtx = torch.zeros((seq_num, self.x_dim, seq_len))
                obs_mtx = torch.zeros((seq_num, self.y_dim, seq_len))
                x_prev = torch.zeros((seq_num, self.x_dim, 1))
                with torch.no_grad():
                    for i in range(seq_len):
                        xt = self.f(x_prev)
                        x_mean = torch.zeros(seq_num, self.x_dim)
                        distrib = MultivariateNormal(loc=x_mean, covariance_matrix=cov_q)
                        eq = distrib.rsample().view(seq_num, self.x_dim, 1)
                        xt = torch.add(xt, eq)
                        yt = self.g(xt)
                        y_mean = torch.zeros(seq_num, self.y_dim)
                        distrib = MultivariateNormal(loc=y_mean, covariance_matrix=cov_r)
                        er = distrib.rsample().view(seq_num, self.y_dim, 1)
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
            if mode == "train":
                self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch

    def f(self, x):
        batched_F = self.F.view(1, self.F.shape[0], self.F.shape[1]).expand(x.shape[0], -1, -1).to(x.device)
        return torch.bmm(batched_F, x)

    def f_single(self, x):
        return torch.matmul(self.F, x)

    def g(self, x):
        if self.is_linear:
            return x.reshape((x.shape[0], x.shape[1], 1))
        else:
            y1 = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
            y2 = torch.arctan2(x[:, 1], x[:, 0])
            return torch.cat((y1, y2), dim=1).reshape((x.shape[0], x.shape[1], 1)).to(x.device)

    def g_single(self, x):
        if self.is_linear:
            return x.reshape((-1, 1))
        else:
            x = x.squeeze()
            y1 = torch.sqrt(x[0] ** 2 + x[1] ** 2)
            y2 = torch.arctan2(x[1], x[0])
            return torch.stack([y1, y2]).reshape((-1, 1))

    def Jacobian_f(self, x, is_seq=True):
        if is_seq:
            return self.F.reshape((1, self.F.shape[0], self.F.shape[1])).repeat(x.shape[0], 1, 1).to(x.device)
        else:
            return self.F.to(x.device)

    def Jacobian_g(self, x, is_seq=True):
        if self.is_linear:
            if is_seq:
                return torch.eye(2).reshape((1, self.y_dim, self.x_dim)).repeat(x.shape[0], 1, 1).to(x.device)
            else:
                return torch.tensor([[1., 0.], [0., 1.]]).to(x.device)
        else:
            if is_seq:
                H11 = x[:, 0] / torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
                H12 = x[:, 1] / torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
                H21 = -x[:, 1] / (x[:, 0] ** 2 + x[:, 1] ** 2)
                H22 = x[:, 0] / (x[:, 0] ** 2 + x[:, 1] ** 2)
                temp = torch.zeros((x.shape[0], self.y_dim, self.x_dim)).to(x.device)
                temp[:, 0, 0] = H11.squeeze()
                temp[:, 0, 1] = H12.squeeze()
                temp[:, 1, 0] = H21.squeeze()
                temp[:, 1, 1] = H22.squeeze()
                return temp
            else:
                H11 = x[0] / torch.sqrt(x[0] ** 2 + x[1] ** 2)
                H12 = x[1] / torch.sqrt(x[0] ** 2 + x[1] ** 2)
                H21 = -x[1] / (x[0] ** 2 + x[1] ** 2)
                H22 = x[0] / (x[0] ** 2 + x[1] ** 2)
                return torch.tensor([[H11, H12], [H21, H22]]).to(x.device)

    def get_H(self):
        return self.H
