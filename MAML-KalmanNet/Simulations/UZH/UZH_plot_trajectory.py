import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt
from filter import Filter
from UZH_syntheticNShot import SyntheticNShot

argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number for danse', default=1000)
argparser.add_argument('--n_way', type=int, help='n way', default=5)
argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
argparser.add_argument('--k_spt_test', type=int, help='k shot for support set test', default=8)
argparser.add_argument('--q_qry', type=int, help='q shot for query set', default=1)
argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=16)
argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.001)
argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=8)
argparser.add_argument('--batch_size', type=int, help='batch size for train MAML', default=6)
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=8)
argparser.add_argument('--task_num_train', type=int, help='task num for train', default=25)
argparser.add_argument('--seq_len', type=int, help='seq length for task', default=80)
argparser.add_argument('--use_cuda', type=int, help='use GPU to accelerate training', default=False)

args = argparser.parse_args()

if args.use_cuda:
    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device('cuda')
    else:
        raise Exception("No GPU found, please set args.use_cuda = False")
else:
    print("Using CPU")
    device = torch.device('cpu')

base_dict = torch.load('../../MAML_model/UZH/basenet.pt')
data_path = '../../MAML_data/UZH/'
model = SyntheticNShot(batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_shot_test=args.k_spt_test,
                       q_query=args.q_qry,
                       data_path=data_path,
                       use_cuda=args.use_cuda,
                       Is_GenData=False)
my_filter = Filter(args, model)
loss_fn = torch.nn.MSELoss()

torch.manual_seed(333)
np.random.seed(333)

state = np.load('../../MAML_data/UZH/state_circle.npy')
obs = np.load('../../MAML_data/UZH/obs_circle.npy')

state = state[0:9, :]
obs = obs[0:3, :]

state_train, obs_train = state[:, 1020:], obs[:, 1020:]
state_test, obs_test = state[:, :1020], obs[:, :1020]


max_start_index = state_train.shape[1] - args.seq_len

start_indices = np.random.randint(0, max_start_index + 1, size=args.task_num_train)
state_train = np.stack([state_train[:, start:start + args.seq_len] for start in start_indices])
obs_train = np.stack([obs_train[:, start:start + args.seq_len] for start in start_indices])


state_train, obs_train, state_test, obs_test = torch.tensor(state_train, dtype=torch.float32), torch.tensor(obs_train, dtype=torch.float32), \
    torch.tensor(state_test, dtype=torch.float32), torch.tensor(obs_test, dtype=torch.float32)
state_train, obs_train, state_test, obs_test = state_train.to(device), obs_train.to(device), \
    state_test.to(device), obs_test.to(device)
state_test, obs_test = state_test.unsqueeze(0), obs_test.unsqueeze(0)

shuffle_indices = torch.randperm(args.task_num_train)
state_train, obs_train = state_train[shuffle_indices], obs_train[shuffle_indices]

# EKF
q2 = 0.5
r2 = 0.5
cov_q = q2 * torch.tensor([[1/4.*model.dt**4, 1/2.*model.dt**3, 1/2.*model.dt**2],
                           [1/2.*model.dt**3,      model.dt**2,         model.dt],
                           [1/2.*model.dt**2,         model.dt,               1]])
cov_q = torch.kron(torch.eye(3), cov_q)

cov_q = cov_q + torch.eye(model.x_dim) * 1e-8

cov_r = r2 * torch.eye(model.y_dim)
loss_ekf = my_filter.EKF(state_test, obs_test, cov_q, cov_r)
state_EKF = my_filter.state_EKF.squeeze()
print('q2 = r2 = 0.2, EKF loss(dB): ' + str(loss_ekf))

losses_dB = []

my_filter.train_net.load_state_dict(base_dict)

for k in range(args.update_step_test):
    loss = my_filter.compute_x_post(state_train, obs_train)
    my_filter.train_optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(my_filter.train_net.parameters(), 1)
    my_filter.train_optim.step()
    with torch.no_grad():
        temp_loss = my_filter.compute_x_post_qry(state_test, obs_test)
        losses_dB.append(10 * torch.log10(temp_loss))
print('q2 = r2 = 0.2, MAML-KalmanNet loss(dB): ' + str(losses_dB[-1]))
state_predict = my_filter.state_predict.squeeze()

indices = [0, 3, 6]
position_hat = state_predict[indices, 1:].detach()
position = state_test.squeeze()[indices, 1:].detach()
position_EKF = state_EKF[indices, 1:].detach()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(position[0], position[1], position[2], label='Ground truth', linewidth=2, color='#808080', linestyle='solid')
ax.plot(position_EKF[0], position_EKF[1], position_EKF[2], label='KF', linewidth=2, color='red', linestyle='dotted')
ax.plot(position_hat[0], position[1], position[2], label='MAML-KalmanNet', linewidth=2, color='#1F77B4', linestyle='dashdot')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.legend()

plt.show()
