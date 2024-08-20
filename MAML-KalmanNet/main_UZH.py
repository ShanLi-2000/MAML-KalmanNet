import torch
import math
import numpy as np
import argparse
from meta import Meta
from Simulations.UZH.UZH_syntheticNShot import SyntheticNShot

def main(args):
    print(args)

    args.use_cuda = True
    if args.use_cuda:
        if torch.cuda.is_available():
            print("Using GPU")
            device = torch.device('cuda')
        else:
            raise Exception("No GPU found, please set args.use_cuda = False")
    else:
        print("Using CPU")
        device = torch.device('cpu')

    path_data = 'MAML_data/UZH/'
    db_train = SyntheticNShot(batchsz=args.task_num,
                              n_way=args.n_way,
                              k_shot=args.k_spt,
                              k_shot_test=args.k_spt_test,
                              q_query=args.q_qry,
                              data_path=path_data,
                              Is_GenData=False,
                              use_cuda=args.use_cuda)

    maml = Meta(args, db_train)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    for step in range(args.epoch):
        state_spt, obs_spt, state_qry, obs_qry, select_num = db_train.next()
        state_spt, obs_spt, state_qry, obs_qry = torch.from_numpy(state_spt), torch.from_numpy(obs_spt), \
            torch.from_numpy(state_qry), torch.from_numpy(obs_qry)
        state_spt, obs_spt, state_qry, obs_qry = state_spt.to(device), obs_spt.to(device), state_qry.to(device), obs_qry.to(device)

        if step <= args.epoch / 2:
            loss_dB, count_num = maml(state_spt, obs_spt, state_qry, obs_qry)
        else:
            loss_dB, count_num = maml.forward_second(state_spt, obs_spt, state_qry, obs_qry)

        if step % 10 == 0:
            print('step:' + str(step) + ' loss_dB: ' + str(loss_dB) + ' count_num ' + str(count_num))

        if step % 1000 == 0 and step != 0:
            torch.save(maml.base_net.state_dict(), './MAML_model/UZH/basenet_' + str(step) + '.pt')

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=8001)
    argparser.add_argument('--n_way', type=int, help='n way', default=6)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=6)  # k_spt>=batch_size
    argparser.add_argument('--k_spt_test', type=int, help='k shot for support set test', default=30)
    argparser.add_argument('--q_qry', type=int, help='q shot for query set', default=15)
    argparser.add_argument('--batch_size', type=int, help='batch size for train MAML', default=4)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=16)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=5e-3)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=4)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=6)
    argparser.add_argument('--use_cuda', type=int, help='use GPU to accelerate training', default=False)

    args = argparser.parse_args()

    main(args)
