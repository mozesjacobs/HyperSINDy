import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class LorenzDataset(Dataset):

    def __init__(self, args, data_path):
        data = np.load(data_path, allow_pickle=True).item()
        self.x = torch.tensor(data['x']).view(-1, args.timesteps, args.u_dim)
        self.dx = torch.tensor(data['dx']).view(-1, args.timesteps, args.u_dim)
        self.dz = torch.tensor(data['dz']).view(-1, args.timesteps, args.z_dim)

    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx], self.dz[idx]


class Lorenz2Dataset(Dataset):

    def __init__(self, args, data_path):
        data = np.load(data_path, allow_pickle=True).item()
        x1 = torch.tensor(data['x1']).view(-1, args.timesteps, args.u_dim)
        x2 = torch.tensor(data['x2']).view(-1, args.timesteps, args.u_dim)
        z = torch.tensor(data['z']).view(-1, args.timesteps, args.z_dim)

        self.x1, self.x2, self.z = [], [], []
        self.x1_next, self.x2_next, self.z_next = [], [], []
        for i in range(args.timesteps):
            if i % args.subsample_rate == 0:
                next_idx = args.tau + i
                if next_idx < args.timesteps:
                    self.x1_next.append(x1[:, next_idx])
                    self.x2_next.append(x2[:, next_idx])
                    self.z_next.append(z[:, next_idx])
                    self.x1.append(x1[:, i])
                    self.x2.append(x2[:, i])
                    self.z.append(z[:, i])
        self.x1 = torch.stack(self.x1, dim=1)
        self.x2 = torch.stack(self.x2, dim=1)
        self.z = torch.stack(self.z, dim=1)
        if args.dynamic:
            self.x1_next = torch.stack(self.x1_next, dim=1)
            self.x2_next = torch.stack(self.x2_next, dim=1)
            self.z_next = torch.stack(self.z_next, dim=1)
        else:
            self.x1_next = self.x1
            self.x2_next = self.x2
            self.z_next = self.x3

    def __len__(self):
        return self.x1.size(0)
    
    def __getitem__(self, idx):
        return (self.x1[idx], self.x1_next[idx], self.x2[idx], self.x2_next[idx]), (self.z[idx], self.z_next[idx])


class Lorenz3Dataset(Dataset):

    def __init__(self, args, data_path):
        data = np.load(data_path, allow_pickle=True).item()
        self.x1 = torch.tensor(data['x1']).view(-1, args.timesteps, args.u_dim)
        self.x2 = torch.tensor(data['x2']).view(-1, args.timesteps, args.u_dim)
        self.x3 = torch.tensor(data['x3']).view(-1, args.timesteps, args.u_dim)
        self.z = torch.tensor(data['z']).view(-1, args.timesteps, args.z_dim)

    def __len__(self):
        return self.x1.size(0)
    
    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.x3[idx], self.z[idx]


class Lorenz4Dataset(Dataset):

    def __init__(self, args, data_path):
        data = np.load(data_path, allow_pickle=True).item()
        self.x1 = torch.tensor(data['x1']).view(-1, args.timesteps, args.u_dim)
        self.x2 = torch.tensor(data['x2']).view(-1, args.timesteps, args.u_dim)
        self.x3 = torch.tensor(data['x3']).view(-1, args.timesteps, args.u_dim)
        self.x4 = torch.tensor(data['x4']).view(-1, args.timesteps, args.u_dim)
        self.z = torch.tensor(data['z']).view(-1, args.timesteps, args.z_dim)

    def __len__(self):
        return self.x1.size(0)
    
    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.x3[idx], self.x4[idx], self.z[idx]


class Lorenz5Dataset(Dataset):

    def __init__(self, args, data_path):
        data = np.load(data_path, allow_pickle=True).item()
        x1 = torch.tensor(data['x1']).view(-1, args.timesteps, args.u_dim)
        x2 = torch.tensor(data['x2']).view(-1, args.timesteps, args.u_dim)
        x3 = torch.tensor(data['x3']).view(-1, args.timesteps, args.u_dim)
        z = torch.tensor(data['z']).view(-1, args.timesteps, args.z_dim)

        self.x1, self.x2, self.x3, self.z = [], [], [], []
        self.x1_next, self.x2_next, self.x3_next, self.z_next = [], [], [], []
        for i in range(args.timesteps):
            if i % args.subsample_rate == 0:
                next_idx = args.tau + i
                if next_idx < args.timesteps:
                    self.x1_next.append(x1[:, next_idx])
                    self.x2_next.append(x2[:, next_idx])
                    self.x3_next.append(x3[:, next_idx])
                    self.z_next.append(z[:, next_idx])
                    self.x1.append(x1[:, i])
                    self.x2.append(x2[:, i])
                    self.x3.append(x3[:, i])
                    self.z.append(z[:, i])
        self.x1 = torch.stack(self.x1, dim=1)
        self.x2 = torch.stack(self.x2, dim=1)
        self.x3 = torch.stack(self.x3, dim=1)
        self.z = torch.stack(self.z, dim=1)
        if args.dynamic:
            self.x1_next = torch.stack(self.x1_next, dim=1)
            self.x2_next = torch.stack(self.x2_next, dim=1)
            self.x3_next = torch.stack(self.x3_next, dim=1)
            self.z_next = torch.stack(self.z_next, dim=1)
        else:
            self.x1_next = self.x1
            self.x2_next = self.x2
            self.x3_next = self.x3
            self.z_next = self.x4

    def __len__(self):
        return self.x1.size(0)
    
    def __getitem__(self, idx):
        return (self.x1[idx], self.x1_next[idx], self.x2[idx], self.x2_next[idx], self.x3[idx], self.x3_next[idx]), (self.z[idx], self.z_next[idx])

class LorenzSimpleDataset(Dataset):

    def __init__(self, args, data_path, z_mean=None, z_std=None):
        data = np.load(data_path, allow_pickle=True).item()
        self.num_comps = args.num_comps
        self.k = args.hankel_trajectory_length

        self.tau = args.tau
        self.z = torch.tensor(data['z'])[0]

        self.z_mean = z_mean
        self.z_std = z_std
        if self.z_mean is None:
            self.z_mean = torch.mean(self.z, dim=0)
            self.z_std = torch.std(self.z, dim=0)
        if args.norm_data:
            self.z = (self.z - self.z_mean) / self.z_std


        self.z = self.z * args.scale_data
        self.z_next = self.z[self.tau:]
        self.z = self.z[:-self.tau]

    def __len__(self):
        return self.z.size(0)
    
    def __getitem__(self, idx):
        return (), (self.z[idx], self.z_next[idx])


class LorenzSimpleDerivativeDataset(Dataset):

    def __init__(self, args, data_path):
        data = np.load(data_path, allow_pickle=True).item()
        self.num_comps = args.num_comps
        self.k = args.hankel_trajectory_length

        self.tau = args.tau
        self.z = torch.tensor(data['z'])[0]
        self.dz = torch.tensor(data['dz'])[0]
        self.z_next = self.z[self.tau:]
        self.z = self.z[:-self.tau]
        self.dz = self.dz[:-self.tau]

    def __len__(self):
        return self.z.size(0)
    
    def __getitem__(self, idx):
        return (), (self.z[idx], self.dz[idx], self.z_next[idx])


class DelayLorenzDataset(Dataset):

    def __init__(self, args, data_path,
                 U_x1=None, U_x2=None, U_z=None,
                 x1_mean=None, x2_mean=None, z_mean=None,
                 x1_std=None, x2_std=None, z_std=None):
        data = np.load(data_path, allow_pickle=True).item()
        self.num_comps = args.num_comps
        self.k = args.hankel_trajectory_length
        self.scale = args.scale_data
        self.data_idx = args.data_idx
            
        x1 = torch.tensor(data['x1']).reshape(-1, args.timesteps, 10)[self.data_idx]
        x2 = torch.tensor(data['x2']).reshape(-1, args.timesteps, 10)[self.data_idx]
        z = torch.tensor(data['z']).reshape(-1, args.timesteps, 3)[self.data_idx]

        m = args.timesteps - self.k
        x1h = hankel_matrix(x1, m=m, d=1)
        x2h = hankel_matrix(x2, m=m, d=1)
        zh = hankel_matrix(z, m=m, d=1)

        if U_x1 is None or U_x2 is None or U_z is None:
            self.x1, self.U_x1 = get_svd(x1h, self.num_comps)
            self.x2, self.U_x2 = get_svd(x2h, self.num_comps)
            self.z, self.U_z = get_svd(zh, self.num_comps)
        else:
            # U_x1 = 10 x 20 x 20
            # x1h = 20 x 980 x 10
            self.U_x1, self.U_x2, self.U_z = U_x1, U_x2, U_z
            x1h = torch.transpose(x1h, 0, 2) # 10 x 980 x 20
            x2h = torch.transpose(x2h, 0, 2) # 10 x 980 x 20
            zh = torch.transpose(zh, 0, 2) # 3 x 980 x 20
            self.x1 = torch.matmul(x1h, self.U_x1)[:, :, 0:args.num_comps]
            self.x2 = torch.matmul(x2h, self.U_x2)[:, :, 0:args.num_comps]
            self.z = torch.matmul(zh, self.U_z)[:, :, 0:args.num_comps]
            self.x1 = torch.transpose(torch.transpose(self.x1, 0, 1), 1, 2) # num, comp, dim
            self.x2 = torch.transpose(torch.transpose(self.x2, 0, 1), 1, 2) # num, comp, dim
            self.z = torch.transpose(torch.transpose(self.z, 0, 1), 1, 2) # num, comp, dim

        # normalize
        self.x1_mean, self.x2_mean, self.z_mean = x1_mean, x2_mean, z_mean
        self.x1_std, self.x2_std, self.z_std = x1_std, x2_std, z_std
        if self.x1_mean is None:
            self.x1_mean = torch.mean(self.x1, dim=0)
            self.x2_mean = torch.mean(self.x2, dim=0)
            self.z_mean = torch.mean(self.z, dim=0)
            self.x1_std = torch.std(self.x1, dim=0)
            self.x2_std = torch.std(self.x2, dim=0)
            self.z_std = torch.std(self.z, dim=0)
        if args.norm_data:
            self.x1 = (self.x1 - self.x1_mean) / self.x1_std
            self.x2 = (self.x2 - self.x2_mean) / self.x2_std
            self.z = (self.z - self.z_mean) / self.z_std
        
        # scale data
        self.x1 = self.x1 * self.scale
        self.x2 = self.x2 * self.scale
        self.z = self.z * self.scale

        # split into curr / next
        tau = args.tau
        if args.dynamic:
            self.x1_next = self.x1[tau:]
            self.x2_next = self.x2[tau:]
            self.z_next = self.z[tau:]
            self.x1 = self.x1[0:-tau]
            self.x2 = self.x2[0:-tau]
            self.z = self.z[0:-tau]
        else:
            self.x1_next = self.x1
            self.x2_next = self.x2
            self.z_next = self.z

    def __len__(self):
        return self.x1.size(0)
    
    def __getitem__(self, idx):
        return (self.x1[idx], self.x1_next[idx], self.x2[idx], self.x2_next[idx]), (self.z[idx], self.z_next[idx])


class DelayLorenzFlexDataset(Dataset):

    def __init__(self, args, data_path,
                 U_x1=None, U_x2=None, U_z=None,
                 x1_mean=None, x2_mean=None, z_mean=None,
                 x1_std=None, x2_std=None, z_std=None):
        data = np.load(data_path, allow_pickle=True).item()
        self.num_comps = args.num_comps
        self.k = args.hankel_trajectory_length
        self.scale = args.scale_data
        self.data_idx = args.data_idx
            
        x1 = torch.tensor(data['x1']).reshape(-1, args.timesteps, 10)[self.data_idx]
        x2 = torch.tensor(data['x2']).reshape(-1, args.timesteps, 10)[self.data_idx]
        z = torch.tensor(data['z']).reshape(-1, args.timesteps, 3)[self.data_idx]

        m = args.timesteps - self.k
        x1h = hankel_matrix(x1, m=m, d=1)
        x2h = hankel_matrix(x2, m=m, d=1)
        zh = hankel_matrix(z, m=m, d=1)

        if U_x1 is None or U_x2 is None or U_z is None:
            self.x1, self.U_x1 = get_svd(x1h, self.num_comps)
            self.x2, self.U_x2 = get_svd(x2h, self.num_comps)
            self.z, self.U_z = get_svd(zh, self.num_comps)
        else:
            # U_x1 = 10 x 20 x 20
            # x1h = 20 x 980 x 10
            self.U_x1, self.U_x2, self.U_z = U_x1, U_x2, U_z
            x1h = torch.transpose(x1h, 0, 2) # 10 x 980 x 20
            x2h = torch.transpose(x2h, 0, 2) # 10 x 980 x 20
            zh = torch.transpose(zh, 0, 2) # 3 x 980 x 20
            self.x1 = torch.matmul(x1h, self.U_x1)[:, :, 0:args.num_comps]
            self.x2 = torch.matmul(x2h, self.U_x2)[:, :, 0:args.num_comps]
            self.z = torch.matmul(zh, self.U_z)[:, :, 0:args.num_comps]
            self.x1 = torch.transpose(torch.transpose(self.x1, 0, 1), 1, 2) # num, comp, dim
            self.x2 = torch.transpose(torch.transpose(self.x2, 0, 1), 1, 2) # num, comp, dim
            self.z = torch.transpose(torch.transpose(self.z, 0, 1), 1, 2) # num, comp, dim

        # normalize
        self.x1_mean, self.x2_mean, self.z_mean = x1_mean, x2_mean, z_mean
        self.x1_std, self.x2_std, self.z_std = x1_std, x2_std, z_std
        if self.x1_mean is None:
            self.x1_mean = torch.mean(self.x1, dim=0)
            self.x2_mean = torch.mean(self.x2, dim=0)
            self.z_mean = torch.mean(self.z, dim=0)
            self.x1_std = torch.std(self.x1, dim=0)
            self.x2_std = torch.std(self.x2, dim=0)
            self.z_std = torch.std(self.z, dim=0)
        if args.norm_data:
            self.x1 = (self.x1 - self.x1_mean) / self.x1_std
            self.x2 = (self.x2 - self.x2_mean) / self.x2_std
            self.z = (self.z - self.z_mean) / self.z_std
        
        # scale data
        self.x1 = self.x1 * self.scale
        self.x2 = self.x2 * self.scale
        self.z = self.z * self.scale

        # split into curr / next
        tau = args.tau
        if args.dynamic:
            self.x1_next_next = self.x1[tau:]
            self.x2_next_next = self.x2[tau:]
            self.z_next_next = self.z[tau:]
            self.x1_next = self.x1[1:]
            self.x2_next = self.x2[1:]
            self.z_next = self.z[1:]
            self.x1 = self.x1[0:-tau]
            self.x2 = self.x2[0:-tau]
            self.z = self.z[0:-tau]
        else:
            self.x1_next = self.x1
            self.x2_next = self.x2
            self.z_next = self.z

    def __len__(self):
        return self.x1.size(0)
    
    def __getitem__(self, idx):
        return (self.x1[idx], self.x1_next[idx], self.x1_next_next[idx], self.x2[idx], self.x2_next[idx], self.x2_next_next[idx]), (self.z[idx], self.z_next[idx], self.z_next_next[idx])


class DelayLorenzSimpleDataset(Dataset):

    def __init__(self, args, data_path, U_z=None, norm_mean=None, norm_std=None):
        data = np.load(data_path, allow_pickle=True).item()
        self.num_comps = args.num_comps
        self.k = args.hankel_trajectory_length
        self.scale = args.scale_data

        z = torch.tensor(data['z'])[0]
        T, z_dim = z.shape

        m = T - self.k
        zh = hankel_matrix(z, m=m, d=1)

        if U_z is None:
            self.z, self.U_z = get_svd(zh, self.num_comps)
        else:
            # U_x1 = 10 x 20 x 20
            # x1h = 20 x 980 x 10
            self.U_z = U_z
            zh = torch.transpose(zh, 0, 2) # dim x num x comp
            self.z = torch.matmul(zh, self.U_z)[:, :, 0:args.num_comps]
            self.z = torch.transpose(torch.transpose(self.z, 0, 1), 1, 2) # num, comp, dim
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        if self.norm_mean is None or self.norm_std is None:
            self.norm_mean = torch.mean(self.z, dim=0)
            self.norm_std = torch.std(self.z, dim=0)
        if args.norm_data:
            self.z = (self.z - self.norm_mean) / self.norm_std
        tau = args.tau
        if args.dynamic:
            self.z_next = self.z[tau:]
            self.z = self.z[0:-tau]
        else:
            self.z_next = self.z
        self.z = self.z * self.scale
        self.z_next = self.z_next * self.scale

    def __len__(self):
        return self.z.size(0)
    
    def __getitem__(self, idx):
        return (), (self.z[idx], self.z_next[idx])


class DelayLorenzSimple2Dataset(Dataset):

    def __init__(self, args, data_path, U_z=None):
        data = np.load(data_path, allow_pickle=True).item()
        self.num_comps = args.num_comps
        self.k = args.hankel_trajectory_length

        z = torch.tensor(data['z'])[0]
        T, z_dim = z.shape

        m = T - self.k
        zh = hankel_matrix(z, m=m, d=1)

        if U_z is None:
            self.z, self.U_z = get_svd(zh, self.num_comps)
        else:
            # U_x1 = 10 x 20 x 20
            # x1h = 20 x 980 x 10
            self.U_z = U_z
            zh = torch.transpose(zh, 0, 2) # 3 x 980 x 20
            self.z = torch.matmul(zh, self.U_z)[:, :, 0:args.num_comps]
            self.z = torch.transpose(torch.transpose(self.z, 0, 1), 1, 2) # num, comp, dim
        tau = args.tau
        num = 25
        if args.dynamic:
            #self.z_next2 = self.z.reshape(-1, 10 * tau, 3)
            #print(self.z_next2.shape)
            self.z_next2 = []
            for i in range(len(self.z) - num):
                self.z_next2.append(self.z[i + 1:num + i + 1])
            self.z_next2 = torch.stack(self.z_next2, dim=0)
            self.z_next = self.z[tau:]
            self.z = self.z[0:-num * tau]
        else:
            self.z_next = self.z
            self.z_next2 = self.z

    def __len__(self):
        return self.z.size(0)
    
    def __getitem__(self, idx):
        return (), (self.z[idx], self.z_next[idx], self.z_next2[idx])


class Arousal1Dataset(Dataset):

    #def __init__(self, brain_path, pupil_path):
    #    self.brain = np.load(brain_path)
    #    self.pupil = np.load(pupil_path)

    def __init__(self, brain, pupil):
        self.brain = torch.from_numpy(brain)
        self.pupil = torch.from_numpy(pupil)

    def __len__(self):
        return self.brain.size(0)
    
    def __getitem__(self, idx):
        return self.brain[idx], self.pupil[idx]

def hankel_matrix(X, m=225, d=1):
    emb = []
    for i in range(m):
        emb.append(X[i*d:X.size(0) - d*(m - i)])
    emb = torch.stack(emb, dim=0)
    return torch.transpose(emb, 0, 1)

def get_svd(X, num_comps=3):
    x1h = torch.transpose(torch.transpose(X, 1, 2), 0, 1)
    U, S, Vh = torch.linalg.svd(x1h, full_matrices=True)
    V = torch.transpose(Vh, 1, 2)[:, :, 0:num_comps] # latdim, samples, comps
    V = torch.transpose(torch.transpose(V, 0, 1), 1, 2)
    return V, U # samples, comps, latdim


class HankelLorenzDataset(Dataset):

    def __init__(self, args, data_path,
                 x1_mean=None, x2_mean=None, z_mean=None,
                 x1_std=None, x2_std=None, z_std=None):
        data = np.load(data_path, allow_pickle=True).item()
        self.num_comps = args.num_comps
        self.k = args.hankel_trajectory_length
        self.scale = args.scale_data
        self.data_idx = args.data_idx
            
        self.x1 = torch.tensor(data['x1']).reshape(-1, args.timesteps, 10)[self.data_idx]
        self.x2 = torch.tensor(data['x2']).reshape(-1, args.timesteps, 10)[self.data_idx]
        self.z = torch.tensor(data['z']).reshape(-1, args.timesteps, 3)[self.data_idx]

        #m = args.timesteps - self.k
        #self.x1 = hankel_matrix(x1, m=m, d=1)[0]
        #self.x2 = hankel_matrix(x2, m=m, d=1)[0]
        #self.z = hankel_matrix(z, m=m, d=1)[0]

        #self.x1 = self.x1[:,4]
        #self.x2 = self.x2[:,4]
        #self.z = self.z[:,4]

        # normalize
        self.x1_mean, self.x2_mean, self.z_mean = x1_mean, x2_mean, z_mean
        self.x1_std, self.x2_std, self.z_std = x1_std, x2_std, z_std
        if self.x1_mean is None:
            self.x1_mean = torch.mean(self.x1, dim=0)
            self.x2_mean = torch.mean(self.x2, dim=0)
            self.z_mean = torch.mean(self.z, dim=0)
            self.x1_std = torch.std(self.x1, dim=0)
            self.x2_std = torch.std(self.x2, dim=0)
            self.z_std = torch.std(self.z, dim=0)
        if args.norm_data:
            self.x1 = (self.x1 - self.x1_mean) / self.x1_std
            self.x2 = (self.x2 - self.x2_mean) / self.x2_std
            self.z = (self.z - self.z_mean) / self.z_std



        m = args.timesteps - self.k
        #self.x1 = hankel_matrix(self.x1, m=m, d=1)[:, :, 0]#[0]
        #self.x2 = hankel_matrix(self.x2, m=m, d=1)[:, :, 0]#[0]
        #self.z = hankel_matrix(self.z, m=m, d=1)#[:, :, 0]#[0]
        #print(self.z.shape)
        #print(self.x1.shape)

        #self.x1 = torch.transpose(self.x1, 0, 1).reshape(self.x1.size(0), -1)
        #self.x2 = torch.transpose(self.x2, 0, 1).reshape(self.x2.size(0), -1)
        #self.z = torch.transpose(self.z, 0, 1)

        self.x1 = hankel_matrix(self.x1, m=m, d=1)#[:, :, 0]#[0]
        self.x2 = hankel_matrix(self.x2, m=m, d=1)#[:, :, 0]#[0]
        self.z = hankel_matrix(self.z, m=m, d=1)#[:, :, 0]#[0]

        self.x1 = torch.transpose(self.x1, 0, 1)#.reshape(self.x1.size(0), -1)
        self.x2 = torch.transpose(self.x2, 0, 1)#.reshape(self.x2.size(0), -1)
        self.z = torch.transpose(self.z, 0, 1)
        
        self.x1 = self.x1.reshape(self.x1.size(0), -1)
        self.x2 = self.x2.reshape(self.x2.size(0), -1)
        #self.z = torch.transpose(self.z, 0, 1)

        #print(self.x1.shape)
        
        # scale data
        self.x1 = self.x1 * self.scale
        self.x2 = self.x2 * self.scale
        self.z = self.z * self.scale

        # split into curr / next
        tau = args.tau
        if args.dynamic:
            self.x1_next = self.x1[tau:]
            self.x2_next = self.x2[tau:]
            self.z_next = self.z[tau:]
            self.x1 = self.x1[0:-tau]
            self.x2 = self.x2[0:-tau]
            self.z = self.z[0:-tau]
        else:
            self.x1_next = self.x1
            self.x2_next = self.x2
            self.z_next = self.z

    def __len__(self):
        return self.x1.size(0)
    
    def __getitem__(self, idx):
        return (self.x1[idx], self.x1_next[idx], self.x2[idx], self.x2_next[idx]), (self.z[idx], self.z_next[idx])


class PendulumDataset(Dataset):

    def __init__(self, args, data_path, the_mean=None, the_std=None):
        data = np.load(data_path, allow_pickle=True).item()
        self.x = torch.tensor(data['x'])
        self.z = torch.tensor(data['z'])
        #print(self.z.shape)
        self.num_samples = self.x.size(0)
        self.timesteps = self.x.size(1)
        self.x_dim = self.x.size(2)

        the_num = 2
        self.x = self.x[0:the_num]
        self.num_samples = the_num

        self.x = self.x.reshape(-1, self.x_dim * self.x_dim)
        self.z = self.z.reshape(-1, 1)

        self.x_mean = the_mean
        self.x_std = the_std
        if self.x_mean is None:
            self.x_mean = torch.mean(self.x, dim=0)
            self.x_std = torch.std(self.x, dim=0)
        if args.norm_data:
            self.x = (self.x - self.x_mean) / self.x_std

        #print(torch.mean(self.x, dim=0))
        #print(torch.std(self.x, dim=0))

        self.x = self.x.reshape(self.num_samples, self.timesteps, -1)
        self.x_next = self.x[:, args.tau:].reshape(-1, self.x_dim * self.x_dim)
        self.x = self.x[:, :-args.tau].reshape(-1, self.x_dim * self.x_dim)
        #print(self.x.shape)
        #print(self.x_next.shape)
        #self.z = self.z.reshape()

    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        return (self.x[idx], self.x_next[idx]), (self.z[idx], )



class StochasticLorenzDataset(Dataset):

    def __init__(self, args, data_path,
                 x1_mean=None, x2_mean=None, z_mean=None,
                 x1_std=None, x2_std=None, z_std=None):
        data = np.load(data_path, allow_pickle=True)
        self.num_comps = args.num_comps
        self.k = args.hankel_trajectory_length
        self.scale = args.scale_data
        self.data_idx = args.data_idx
        m = data.shape[0] - self.k
        self.z = torch.tensor(data)

        # normalize
        self.x1_mean, self.x2_mean, self.z_mean = x1_mean, x2_mean, z_mean
        self.x1_std, self.x2_std, self.z_std = x1_std, x2_std, z_std
        if self.x1_mean is None:
            self.x1_mean = None
            self.x2_mean = None
            self.z_mean = torch.mean(self.z, dim=0)
            self.x1_std = None
            self.x2_std = None
            self.z_std = torch.std(self.z, dim=0)
        if args.norm_data:
            self.z = (self.z - self.z_mean) / self.z_std
        self.z = hankel_matrix(self.z, m=m, d=1)
        self.z = torch.transpose(self.z, 0, 1)

        # x1 and x2
        self.x1 = self.z[:, :, 0]
        self.x2 = self.z[:, :, 1]

        # scale data
        self.x1 = self.x1 * self.scale
        self.x2 = self.x2 * self.scale
        self.z = self.z * self.scale

        # split into curr / next
        tau = args.tau
        if args.dynamic:
            self.x1_next = self.x1[tau:]
            self.x2_next = self.x2[tau:]
            self.z_next = self.z[tau:]
            self.x1 = self.x1[0:-tau]
            self.x2 = self.x2[0:-tau]
            self.z = self.z[0:-tau]
        else:
            self.x1_next = self.x1
            self.x2_next = self.x2
            self.z_next = self.z

    def __len__(self):
        return self.x1.size(0)
    
    def __getitem__(self, idx):
        return (self.x1[idx], self.x1_next[idx], self.x2[idx], self.x2_next[idx]), (self.z[idx], self.z_next[idx])



class StochasticLorenzFlexDataset(Dataset):

    def __init__(self, args, data_path,
                 x1_mean=None, x2_mean=None, z_mean=None,
                 x1_std=None, x2_std=None, z_std=None):
        data = np.load(data_path, allow_pickle=True)
        self.num_comps = args.num_comps
        self.k = args.hankel_trajectory_length
        self.scale = args.scale_data
        self.data_idx = args.data_idx
        m = data.shape[0] - self.k
        self.z = torch.tensor(data)

        # normalize
        self.x1_mean, self.x2_mean, self.z_mean = x1_mean, x2_mean, z_mean
        self.x1_std, self.x2_std, self.z_std = x1_std, x2_std, z_std
        if self.x1_mean is None:
            self.x1_mean = None
            self.x2_mean = None
            self.z_mean = torch.mean(self.z, dim=0)
            self.x1_std = None
            self.x2_std = None
            self.z_std = torch.std(self.z, dim=0)
        if args.norm_data:
            self.z = (self.z - self.z_mean) / self.z_std
        self.z = hankel_matrix(self.z, m=m, d=1)
        self.z = torch.transpose(self.z, 0, 1)

        # x1 and x2
        self.x1 = self.z[:, :, 0]
        self.x2 = self.z[:, :, 1]

        # scale data
        self.x1 = self.x1 * self.scale
        self.x2 = self.x2 * self.scale
        self.z = self.z * self.scale

        # split into curr / next
        tau = args.tau
        self.taus = []
        if args.dynamic:
            self.x1_next, self.x2_next, self.z_next = [], [], []
            for i in range(self.z.size(0) - tau):
                curr_tau = np.random.randint(1, tau + 1)
                self.x1_next.append(self.x1[i + curr_tau])
                self.x2_next.append(self.x2[i + curr_tau])
                self.z_next.append(self.z[i + curr_tau])
                self.taus.append(curr_tau)
            self.x1_next = torch.stack(self.x1_next, dim=0)
            self.x2_next = torch.stack(self.x2_next, dim=0)
            self.z_next = torch.stack(self.z_next, dim=0)
            self.taus = torch.tensor(self.taus).unsqueeze(1)
            self.x1 = self.x1[0:-tau]
            self.x2 = self.x2[0:-tau]
            self.z = self.z[0:-tau]
        else:
            self.x1_next = self.x1
            self.x2_next = self.x2
            self.z_next = self.z

    def __len__(self):
        return self.x1.size(0)
    
    def __getitem__(self, idx):
        return (self.x1[idx], self.x1_next[idx], self.x2[idx], self.x2_next[idx]), (self.z[idx], self.z_next[idx]), self.taus[idx]