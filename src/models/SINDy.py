import torch
import torch.nn as nn
from src.utils.model_utils import load_batch
from src.models.HyperNet2 import HyperNet


class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()

        # parameters
        self.z_dim = args.z_dim
        self.in_dim_1 = args.in_dim_1
        self.in_dim_2 = args.in_dim_2
        self.hidden_dim = args.hidden_dim
        self.embedding_dim = args.embedding_dim
        self.data_set = args.data_set

        self.noise_dim = 25

        # hypernets
        self.h1_w = HyperNet(self.noise_dim, (64, 3))
        self.h1_b = HyperNet(self.noise_dim, (64, ))
        self.h2_w = HyperNet(self.noise_dim, (64, 64))
        self.h2_b = HyperNet(self.noise_dim, (64, ))
        self.h3_w = HyperNet(self.noise_dim, (3, 64))
        self.h3_b = HyperNet(self.noise_dim, (3, ))
        
        # reparameterize layers for vae 1
        self.fc_mu = nn.Linear(self.in_dim_1, self.in_dim_1)
        
        # reconstruction loss function
        self.recon_loss = nn.MSELoss(reduction='none')
        self.elu = nn.ELU()
    

    def forward(self, batch, just_mean=False):
        # setup
        device = self.fc_mu.weight.device
        (_, _, _, _), (z, z_next) = load_batch(self.data_set, batch, device)
        batch_size = z.size(0)

        # generate decoder
        n = torch.randn((batch_size, self.noise_dim), device=device)
        #n = z
        self.sample_decoder(n)
        z_hat = self.decode(z)
        recon = self.recon_loss(z_hat, z_next).sum(1).mean()
        #recon = self.recon_loss(z_hat, z_next).mean()
        kld = self.kl(num_samples=batch_size)
        #kld = self.kl(num_samples=200)
        return ([recon], kld), (_, _, z_hat)

    def sample_decoder(self, n=None, batch_size=1, device='cpu'):
        self.w1 = self.h1_w(n, batch_size, device=device)
        self.b1 = self.h1_b(n, batch_size, device=device)
        self.w2 = self.h2_w(n, batch_size, device=device)
        self.b2 = self.h2_b(n, batch_size, device=device)
        self.w3 = self.h3_w(n, batch_size, device=device)
        self.b3 = self.h3_b(n, batch_size, device=device)
        return self.w1, self.b1, self.w2, self.b2, self.w3, self.b3

    def decode(self, z):
        z = torch.bmm(self.w1, z.unsqueeze(2)).squeeze() + self.b1
        z = self.elu(z)
        z = torch.bmm(self.w2, z.unsqueeze(2)).squeeze() + self.b2
        z = self.elu(z)
        z = torch.bmm(self.w3, z.unsqueeze(2)).squeeze() + self.b3
        return z
        
    # calculates mu and logvar
    def posterior(self, e):
        return self.fc_mu(e), self.fc_lv(e)


    # returns a sample from the gaussian distribution given by mu and logvar
    def reparameterize(self, mu, logvar, just_mean=False):
        if just_mean:
            return mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(logvar)
            return mu + eps * std


    # got the kld from the disentangled sequential autoencoder repo
    #def get_kld(self, mu, logvar):
    #    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1).mean()

    def sample_weights_for_kl(self, num_samples=5):
        w1, b1, w2, b2, w3, b3 = self.sample_decoder(batch_size=num_samples, device=self.fc_mu.weight.device)
        w1 = w1.view(num_samples, -1)
        b1 = b1.view(num_samples, -1)
        w2 = w2.view(num_samples, -1)
        b2 = b2.view(num_samples, -1)
        w3 = w3.view(num_samples, -1)
        b3 = b3.view(num_samples, -1)
        #w1 = self.w1.view(num_samples, -1)
        #b1 = self.b1.view(num_samples, -1)
        #w2 = self.w2.view(num_samples, -1)
        #b2 = self.b2.view(num_samples, -1)
        #w3 = self.w3.view(num_samples, -1)
        #b3 = self.b3.view(num_samples, -1)
        gen_weights = torch.cat([w1, b1, w2, b2, w3, b3], 1)
        return gen_weights

    def kl(self, num_samples=5, full_kernel=True):

        gen_weights = self.sample_weights_for_kl(num_samples=num_samples)
        gen_weights = gen_weights.transpose(1, 0)
        prior_samples = torch.randn_like(gen_weights)

        eye = torch.eye(num_samples, device=gen_weights.device)
        wp_distances = (prior_samples.unsqueeze(2) - gen_weights.unsqueeze(1)) ** 2
        # [weights, samples, samples]

        ww_distances = (gen_weights.unsqueeze(2) - gen_weights.unsqueeze(1)) ** 2

        if full_kernel:
            wp_distances = torch.sqrt(torch.sum(wp_distances, 0) + 1e-8)
            wp_dist = torch.min(wp_distances, 0)[0]

            ww_distances = torch.sqrt(
                torch.sum(ww_distances, 0) + 1e-8) + eye * 1e10
            ww_dist = torch.min(ww_distances, 0)[0]

            # mean over samples
            kl = torch.mean(torch.log(wp_dist / (ww_dist + 1e-8) + 1e-8))
            kl *= gen_weights.shape[0]
            #kl += np.log(float(num_samples) / (num_samples - 1))
            kl += torch.log(torch.tensor(float(num_samples) / (num_samples - 1)))
        else:
            wp_distances = torch.sqrt(wp_distances + 1e-8)
            wp_dist = torch.min(wp_distances, 1)[0]

            ww_distances = (torch.sqrt(ww_distances + 1e-8)
                            + (eye.unsqueeze(0) * 1e10))
            ww_dist = torch.min(ww_distances, 1)[0]

            # sum over weights, mean over samples
            kl = torch.sum(torch.mean(
                torch.log(wp_dist / (ww_dist + 1e-8) + 1e-8)
                + torch.log(float(num_samples) / (num_samples - 1)), 1))

        return kl
    

    # returns reconstruction losses and klds
    def vae_loss(self, x1, x1_hat, x2, x2_hat, mu, logvar):
        # reconstruction losses
        recon1 = self.recon_loss(x1_hat, x1).sum(1).mean()
        recon2 = self.recon_loss(x2_hat, x2).sum(1).mean()
        
        # klds
        kld = self.get_kld(mu, logvar)

        return (recon1, recon2), kld
    
    def sample_trajectory(self, T, just_mean=False, z=None, x1=None, x2=None):
        zs = [z]
        self.sample_decoder(batch_size=z.size(0), device=self.fc_mu.weight.device)
        #self.sample_decoder(n=z, device=self.fc_mu.weight.device)
        for t in range(T):
            #self.sample_decoder(batch_size=z.size(0), device=self.fc_mu.weight.device)
            z = self.decode(z)
            zs.append(z)
        return torch.stack(zs, dim=1).detach().cpu().numpy() 