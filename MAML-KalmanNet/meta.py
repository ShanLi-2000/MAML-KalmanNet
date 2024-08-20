import math
import torch
import numpy as np
from torch import nn
from torch import optim
from state_dict_learner import Learner
from filter import Filter

class Meta(nn.Module):

    def __init__(self, args, model, is_linear_net=True):
        super(Meta, self).__init__()
        # super().__init__()
        self.args = args
        self.is_linear_net = is_linear_net
        self.update_lr = args.update_lr  # 0.4
        self.meta_lr = args.meta_lr  # 0.001
        self.n_way = args.n_way  # 5
        self.k_spt = args.k_spt  # 1
        self.k_qry = args.q_qry  # 15
        self.task_num = args.task_num  # 32
        self.update_step = args.update_step  # 5
        self.update_step_test = args.update_step_test  # 10
        if args.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise Exception("No GPU found, please set args.use_cuda = False")
        else:
            self.device = torch.device('cpu')

        self.my_filter = Filter(args, model, is_linear_net=is_linear_net)
        self.model = model
        self.base_net = Learner(self.model.x_dim, self.model.y_dim, args, is_linear_net).to(self.device)

        self.meta_optim = optim.Adam(self.base_net.parameters(), lr=self.meta_lr)
        self.loss_fn = torch.nn.MSELoss()
        self.weight_decay = [0.3, 0.3, 0.2, 0.1, 0.1, 0.01]

    def forward(self, state_spt, obs_spt, state_qry, obs_qry):

        # turn on anomaly detection mode
        # torch.autograd.set_detect_anomaly(True)

        task_num = state_spt.shape[0]
        count_num = task_num
        loss_q = 0
        gradients = {}
        temp_gradients = {}
        losses = torch.tensor(0.).to(self.device)
        for name, param in self.base_net.named_parameters():
            if param.requires_grad:
                gradients[name] = torch.zeros_like(param)
                temp_gradients[name] = torch.zeros_like(param)

        for i in range(task_num):

            task_model = Learner(self.model.x_dim, self.model.y_dim, self.args, self.is_linear_net).to(self.device)
            task_model.load_state_dict(self.base_net.state_dict())
            task_model.initialize_hidden()
            inner_optimizer = optim.SGD(task_model.parameters(), lr=self.update_lr)

            loss = self.my_filter.compute_x_post(state_spt[i], obs_spt[i], task_net=task_model)

            if math.isnan(loss):
                count_num = count_num - 1
                continue

            inner_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(task_model.parameters(), 1)
            inner_optimizer.step()

            for k in range(1, self.update_step):
                loss = self.my_filter.compute_x_post(state_spt[i], obs_spt[i], task_net=task_model)
                if math.isnan(loss) or math.isnan(loss_q):
                    count_num = count_num - 1
                    break
                inner_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(task_model.parameters(), 1)
                inner_optimizer.step()

                loss_q = self.my_filter.compute_x_post_qry(state_qry[i], obs_qry[i], task_net=task_model)
                loss_q.backward()
                torch.nn.utils.clip_grad_norm_(task_model.parameters(), 1)
                for name, param in task_model.named_parameters():
                    if param.grad is not None:
                        temp_gradients[name] += param.grad.clone() * self.weight_decay[k-1]

            if math.isnan(loss) or math.isnan(loss_q):
                for name, param in task_model.named_parameters():
                    if param.grad is not None:
                        temp_gradients[name] = torch.zeros_like(param)
                continue
            else:
                for name, param in task_model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += temp_gradients[name].clone()
                        temp_gradients[name] = torch.zeros_like(param)

            losses += loss_q.clone()

        if count_num == 0:
            return 0, 0

        for param in self.base_net.parameters():
            if param.grad is not None:
                param.grad.zero_()

        for name, param in self.base_net.named_parameters():
            if name in gradients:
                param.grad = gradients[name] / count_num

        torch.nn.utils.clip_grad_norm_(self.base_net.parameters(), 1)
        self.meta_optim.step()
        return 10 * torch.log10(losses/count_num), count_num

    def forward_second(self, state_spt, obs_spt, state_qry, obs_qry):

        # torch.autograd.set_detect_anomaly(True)

        task_num = state_spt.shape[0]
        count_num = task_num
        meta_loss = torch.tensor(0.)
        is_qry_nan = False
        gradients = {}
        for name, param in self.base_net.named_parameters():
            if param.requires_grad:
                gradients[name] = torch.zeros_like(param)

        for i in range(task_num):

            task_model = Learner(self.model.x_dim, self.model.y_dim, self.args, self.is_linear_net).to(self.device)
            task_model.load_state_dict(self.base_net.state_dict())
            task_model.initialize_hidden(is_train=True)
            inner_optimizer = optim.Adam(task_model.parameters(), lr=self.meta_lr)

            loss = self.my_filter.compute_x_post(state_spt[i], obs_spt[i], task_net=task_model)
            if torch.isnan(loss).item():
                count_num -= 1
                continue

            inner_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(task_model.parameters(), 1)
            inner_optimizer.step()

            for k in range(1, self.update_step):

                loss = self.my_filter.compute_x_post(state_spt[i], obs_spt[i], task_net=task_model)
                if torch.isnan(loss).item():
                    count_num -= 1
                    is_qry_nan = True
                    break

                inner_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(task_model.parameters(), 1)
                inner_optimizer.step()

            if is_qry_nan:
                is_qry_nan = False
                continue

            task_model.initialize_hidden(is_train=False)
            loss_qry = self.my_filter.compute_x_post_qry(state_qry[i], obs_qry[i], task_net=task_model)

            meta_loss = meta_loss + loss_qry

        if count_num == 0:
            return 0, 0

        meta_loss = meta_loss / task_num
        meta_loss.backward()

        for name, param in task_model.named_parameters():
            if param.grad is not None:
                gradients[name] += param.grad.clone()

        for param in self.base_net.parameters():
            if param.grad is not None:
                param.grad.zero_()

        for name, param in self.base_net.named_parameters():
            if name in gradients:
                param.grad = gradients[name].clone()

        torch.nn.utils.clip_grad_norm_(self.base_net.parameters(), 1)
        self.meta_optim.step()

        return 10 * torch.log10(meta_loss), count_num
