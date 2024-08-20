import torch
from torch import nn

class Learner(nn.Module):
    def __init__(self, x_dim, y_dim, args, is_linear_net=True):
        super(Learner, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.is_linear_net = is_linear_net
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if self.is_linear_net:
            l1_input = x_dim * 2 + y_dim * 2
        else:
            l1_input = x_dim * 2 + y_dim * 2 + x_dim ** 2 + x_dim * y_dim

        l1_hidden = (x_dim + y_dim) * 10 * 4
        self.l1 = nn.Sequential(
            nn.Linear(l1_input, l1_hidden),
            nn.ReLU()
        )

        self.gru_n_layer = 1
        self.gru_hidden_dim = 4 * (x_dim**2 + y_dim**2)

        self.hn_train_init = torch.randn(self.gru_n_layer, args.batch_size, self.gru_hidden_dim).to(self.device)
        self.hn_qry_init = torch.randn(self.gru_n_layer, args.q_qry, self.gru_hidden_dim).to(self.device)

        self.GRU = nn.GRU(input_size=l1_hidden, hidden_size=self.gru_hidden_dim, num_layers=self.gru_n_layer)

        self.l2 = nn.Sequential(
            nn.Linear(in_features=self.gru_hidden_dim, out_features=x_dim * y_dim * 4),
            nn.ReLU(),
            nn.Linear(in_features=x_dim * y_dim * 4, out_features=x_dim * y_dim)
        )


    def forward(self, state_inno, residual, diff_state, diff_obs, F_Jacobian, H_Jacobian):

        if self.is_linear_net:
            x = torch.cat((state_inno, residual, diff_state, diff_obs), 1).reshape(1, residual.shape[0], -1)
        else:
            x = torch.cat((state_inno, residual, diff_state, diff_obs, F_Jacobian.reshape(residual.shape[0], -1, 1),
                           H_Jacobian.reshape(residual.shape[0], -1, 1)), 1).reshape(1, residual.shape[0], -1)
        l1_out = self.l1(x)
        GRU_in = torch.zeros(1, state_inno.shape[0], (self.x_dim + self.y_dim) * 10 * 4).to(self.device)
        GRU_in[0, :, :] = l1_out.squeeze().clone()
        GRU_out, hn = self.GRU(GRU_in, self.hn.clone())
        self.hn = hn.detach().clone()
        l2_out = self.l2(GRU_out)
        kalman_gain = torch.reshape(l2_out, (state_inno.shape[0], self.x_dim, self.y_dim))

        return kalman_gain

    def initialize_hidden(self, is_train=True):
        if is_train:
            self.hn = self.hn_train_init.detach().clone()
        else:
            self.hn = self.hn_qry_init.detach().clone()
