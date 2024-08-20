import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
from filter import Filter
from linear_syntheticNShot import SyntheticNShot
from state_dict_learner import Learner
from torch.distributions.multivariate_normal import MultivariateNormal

argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number for danse', default=1000)
argparser.add_argument('--n_way', type=int, help='n way', default=5)
argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
argparser.add_argument('--k_spt_test', type=int, help='k shot for support set test', default=8)
argparser.add_argument('--q_qry', type=int, help='q shot for query set', default=15)
argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=16)
argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.00096)
argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=8)
argparser.add_argument('--batch_size', type=int, help='batch size for train MAML', default=4)
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=16)
argparser.add_argument('--task_num_train', type=int, help='task num for train', default=25)
argparser.add_argument('--task_num_test', type=int, help='task num for test', default=15)
argparser.add_argument('--use_cuda', type=int, help='use GPU to accelerate training', default=False)

args = argparser.parse_args()
args.use_cuda = False

if args.use_cuda:
    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device('cuda')
    else:
        raise Exception("No GPU found, please set args.use_cuda = False")
else:
    print("Using CPU")
    device = torch.device('cpu')

Generate_data = True
Train = False
data_path = '../../MAML_data/nonlinear/'
model = SyntheticNShot(batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_shot_test=args.k_spt_test,
                       q_query=args.q_qry,
                       data_path=data_path,
                       use_cuda=args.use_cuda,
                       Is_GenData=False,
                       Is_linear=False)
my_filter = Filter(args, model, is_linear_net=True)
loss_fn = torch.nn.MSELoss()
random_net = Learner(model.x_dim, model.y_dim, args, is_linear_net=True).to(device)
my_dict = torch.load('../../MAML_model/nonlinear/basenet.pt')

torch.manual_seed(333)
np.random.seed(333)

seq_num = 200
seq_len = 50

noise_q = [0.15]
v = [-10., -5., 0., 5., 10., 15., 20.]
noise_num = len(v) * len(noise_q)
if Generate_data:

    state = torch.zeros((noise_num, seq_num, model.x_dim, seq_len))
    obs = torch.zeros((noise_num, seq_num, model.y_dim, seq_len))
    j = 0
    for q2 in noise_q:

        for elem in v:

            cov_q = q2 * torch.eye(model.x_dim)
            r2 = q2 * 10 ** (-elem / 10)
            cov_r = r2 * torch.eye(model.y_dim)
            print("Generating Data: q2: " + str(q2) + "  r2: " + str(r2))
            state_mtx = torch.zeros((seq_num, model.x_dim, seq_len))
            obs_mtx = torch.zeros((seq_num, model.y_dim, seq_len))
            x_prev = torch.zeros((seq_num, model.x_dim, 1))
            with torch.no_grad():
                for i in range(seq_len):
                    xt = model.f(x_prev)
                    x_mean = torch.zeros(seq_num, model.x_dim)
                    distrib = MultivariateNormal(loc=x_mean, covariance_matrix=cov_q)
                    eq = distrib.rsample().view(seq_num, model.x_dim, 1).to(device)
                    xt = torch.add(xt, eq)
                    yt = model.g(xt)
                    y_mean = torch.zeros(seq_num, model.y_dim)
                    distrib = MultivariateNormal(loc=y_mean, covariance_matrix=cov_r)
                    er = distrib.rsample().view(seq_num, model.y_dim, 1).to(device)
                    yt = torch.add(yt, er)
                    x_prev = xt.clone()
                    state_mtx[:, :, i] = torch.squeeze(xt, 2)
                    obs_mtx[:, :, i] = torch.squeeze(yt, 2)

                state[j] = state_mtx
                obs[j] = obs_mtx
                j += 1
    torch.save(state, '../../MAML_data/nonlinear/plot_data/state.pt')
    torch.save(obs, '../../MAML_data/nonlinear/plot_data/obs.pt')
else:
    state = torch.load('../../MAML_data/nonlinear/plot_data/state.pt')
    obs = torch.load('../../MAML_data/nonlinear/plot_data/obs.pt')

for m in range(len(noise_q)):
    selected_task_num = np.random.choice(seq_num, args.task_num_train + args.task_num_test, False)

    state_spt = state[m*len(v):(m+1)*len(v), selected_task_num[:args.task_num_train], :, :]
    obs_spt = obs[m*len(v):(m+1)*len(v), selected_task_num[:args.task_num_train], :, :]
    state_qry = state[m*len(v):(m+1)*len(v), selected_task_num[args.task_num_train:], :, :]
    obs_qry = obs[m*len(v):(m+1)*len(v), selected_task_num[args.task_num_train:], :, :]

    losses_dB_KF = []
    i = 0
    for state_qry_one, obs_qry_one in zip(state_qry, obs_qry):
        q2 = noise_q[0]
        r2 = q2 * 10 ** (-v[i] / 10)
        cov_q = q2 * torch.eye(model.x_dim)
        cov_r = r2 * torch.eye(model.y_dim)
        losses_dB_KF_test = my_filter.EKF(state_qry_one, obs_qry_one, cov_q, cov_r)
        losses_dB_KF.append(losses_dB_KF_test)
        i = i + 1

    print('q2= ' + str(noise_q[m]) + ' KF loss(dB): ' + str(losses_dB_KF))
    state_spt, obs_spt, state_qry, obs_qry = state_spt.to(device), obs_spt.to(device), state_qry.to(device), obs_qry.to(device)

    losses_dB = []
    losses_dB_unsupervised = []
    losses_dB_last = []
    losses_dB_8 = []
    losses_dB_last_unsupervised = []
    losses_dB_8_unsupervised = []

    for i, (state_spt_one, obs_spt_one, state_qry_one, obs_qry_one) in enumerate(zip(state_spt, obs_spt, state_qry, obs_qry)):

        # MAML-KalmanNet -- supervised
        my_filter.train_net.load_state_dict(my_dict)

        for k in range(args.update_step_test):

            loss = my_filter.compute_x_post(state_spt_one, obs_spt_one)
            my_filter.train_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(my_filter.train_net.parameters(), 5)
            my_filter.train_optim.step()

            with torch.no_grad():
                loss_q = my_filter.compute_x_post_qry(state_qry_one, obs_qry_one)
                losses_dB.append(10 * torch.log10(loss_q))

        losses_dB_last.append(losses_dB[-1])
        losses_dB_8.append(losses_dB[-9])

        # MAML-KalmanNet -- unsupervised
        task_model = Learner(model.x_dim, model.y_dim, args, is_linear_net=True).to(device)
        task_model.load_state_dict(my_dict)
        task_model.initialize_hidden(is_train=True)
        inner_optimizer = optim.Adam(task_model.parameters(), lr=args.update_lr)
        my_filter.train_net.load_state_dict(my_dict)

        for k in range(args.update_step_test):
            _ = my_filter.compute_x_post(state_spt_one, obs_spt_one, task_net=task_model)
            loss = nn.MSELoss()(my_filter.temp_y_predict_history[:, :, 1:-1], my_filter.temp_batch_y[:, :, 2:])
            inner_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(task_model.parameters(), 10)
            inner_optimizer.step()

            with torch.no_grad():
                loss_q = my_filter.compute_x_post_qry(state_qry_one, obs_qry_one, task_net=task_model)
                losses_dB_unsupervised.append(10 * torch.log10(loss_q))

        losses_dB_last_unsupervised.append(losses_dB_unsupervised[-1])
        losses_dB_8_unsupervised.append(losses_dB_unsupervised[-9])

    print('q2= ' + str(noise_q[m]) + ' Final update loss: ' + str(losses_dB_last))
    print('q2= ' + str(noise_q[m]) + ' 8 times update loss: ' + str(losses_dB_8))
    print('q2= ' + str(noise_q[m]) + ' Final unsupervised update loss: ' + str(losses_dB_last_unsupervised))
    print('q2= ' + str(noise_q[m]) + ' 8 times unsupervised update loss: ' + str(losses_dB_8_unsupervised))

    plt.plot(v, losses_dB_KF, color='red', label='EKF', linestyle='--',
             marker='^', markerfacecolor='none', markeredgecolor='black', markersize=10)
    # plt.plot(v, losses_dB_8_unsupervised, label='Unsupervised MAML-KalmanNet (train rounds = 8)', marker='H', color='#3E8E8E')
    plt.plot(v, losses_dB_last_unsupervised, label='Unsupervised MAML-KalmanNet (train rounds = 16)', marker='^', color='#413696')
    # plt.plot(v, losses_dB_8, label='MAML-KalmanNet (train rounds = 8)', marker='H', color='#FF7F01')
    plt.plot(v, losses_dB_last, label='MAML-KalmanNet (train rounds = 16)', marker='^', color='#1F77B4')

plt.rcParams['font.family'] = 'Times New Roman'
fontsize = 12
plt.xlabel(r'$V [dB] = 10lg\dfrac{q2}{r2}$', fontsize=fontsize, fontweight='bold')
plt.ylabel('MSE [dB]', fontsize=fontsize)
plt.legend(fontsize=fontsize, loc=3, prop={'size': 10})
plt.grid()
plt.tight_layout()
plt.show()
