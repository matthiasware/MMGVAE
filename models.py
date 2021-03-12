from torch import nn
from torch.nn import functional as F


class EncoderMMG(nn.Module):
    def __init__(self, d_in, d_hidden, d_z, d_t):
        super(EncoderMMG, self).__init__()

        self.d_z = d_z
        self.d_t = d_t

        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
        )
        self.net_means = nn.Sequential(
            nn.Linear(d_hidden, d_z * d_t),
        )
        self.net_logvars = nn.Sequential(
            nn.Linear(d_hidden, d_z * d_t),
        )

        self.net_alphas = nn.Sequential(
            nn.Linear(d_hidden, d_t),
        )

    def forward(self, x):
        inter = self.net(x)
        z_mus = self.net_means(inter)
        z_mus = z_mus.view((-1, self.d_t, self.d_z))

        z_alphas = self.net_alphas(inter)
        z_alphas = F.softmax(z_alphas, dim=1)

        # we predict log_var = log(std**2)
        # -> std = exp(0.5 * log_var)
        # -> alternative is to directly predict std ;)
        z_logvars = self.net_logvars(inter)
        z_logvars = z_logvars.view((-1, self.d_t, self.d_z))

        return z_mus, z_logvars, z_alphas


class EncoderNormal(nn.Module):
    def __init__(self, d_in, d_hidden, d_z):
        super(EncoderNormal, self).__init__()

        self.d_z = d_z

        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
        )
        self.net_mean = nn.Sequential(
            nn.Linear(d_hidden, d_z),
        )
        self.net_logvar = nn.Sequential(
            nn.Linear(d_hidden, d_z),
        )

    def forward(self, x):
        inter = self.net(x)
        z_mu = self.net_mean(inter)

        # we predict log_var = log(std**2)
        # -> std = exp(0.5 * log_var)
        # -> alternative is to directly predict std ;)
        z_logvar = self.net_logvar(inter)

        return z_mu, z_logvar


class Decoder(nn.Module):
    def __init__(self, d_z, d_hidden, d_out):
        super(Decoder, self).__init__()
        #
        self.net = nn.Sequential(
            nn.Linear(d_z, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
