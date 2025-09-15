
import numpy as np
import os
import einops
from scipy.signal import find_peaks
from scipy.linalg import solve 
from scipy.linalg import eig
from numpy.polynomial import Polynomial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Capsule(nn.Module):
    """
    A class implementing capsule layers

    """
    def __init__(self, in_caps_num, in_caps_dim, out_caps_num, out_caps_dim, num_routing=3):
        super(Capsule, self).__init__()
        self.in_caps_num = in_caps_num
        self.in_caps_dim = in_caps_dim
        self.out_caps_num = out_caps_num
        self.out_caps_dim = out_caps_dim
        self.num_routing = num_routing
        self.W = nn.Parameter(0.5 * torch.randn(out_caps_num, in_caps_num, out_caps_dim, in_caps_dim))
        self.b = nn.Parameter(torch.zeros(1, self.out_caps_num, self.in_caps_num))
    
    def squash(self, x):
        """
        """
        L2_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        out = (1 - 1/torch.exp(L2_norm)) * (x / (L2_norm + 1e-8))

        return out
    
    def routing(self, u_hat):
        """
        """
        batch_size = u_hat.shape[0]
        b = self.b.repeat(batch_size, 1, 1)
        
        u_hat_d = u_hat.detach()

        for i in range(self.num_routing):

            c = F.softmax(b, dim=1)

            if i == self.num_routing - 1:
                v = self.squash(torch.sum(c[:, :, :, None] * u_hat, dim=-2, keepdim=True))
            else:
                v = self.squash(torch.sum(c[:, :, :, None] * u_hat_d, dim=-2, keepdim=True))
                b = b + torch.sum(v * u_hat_d, dim=-1)
        
        return v, c
    
    def forward(self, x):
        """
        """
        u_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        v, c = self.routing(u_hat)

        return v.squeeze(dim=-2), c


class PrimaryCapsule(nn.Module):
    """
    A class implementing primary capsule layer.

    """
    def __init__(self, caps_num=16, caps_dim=6*4*4, in_channels=256, out_channels=6, kernel_size=2): 
        super(PrimaryCapsule, self).__init__()
        self.caps_num = caps_num
        self.caps_dim = caps_dim
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
            for _ in range(caps_num)])

    def squash(self, x):
        """
        """
        L2_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        out = (1 - 1/torch.exp(L2_norm)) * (x / (L2_norm + 1e-8))

        return out

    def forward(self, x):
        """
        """
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.flatten(3, 4).flatten(1,2)

        return self.squash(u)

class DeepBeamformer(nn.Module):
    """
    A class implementing CapsFormer

    """
    def __init__(self, num_antennas): 
        super(DeepBeamformer, self).__init__()
        self.num_antennas = num_antennas
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=2),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=2),
            nn.GELU()
        )

        self.prim_caps = PrimaryCapsule()
        
        self.capsule_1 = Capsule(96, 16, 181, 16) 

        self.fc_layers = nn.Sequential(
            nn.Linear(181 * 16, 1024),
            nn.GELU(),
            nn.Linear(1024, 768), 
            nn.GELU(),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256), 
            nn.GELU()
        )

        self.pred_head = nn.Sequential(
            nn.Linear(256, 2 * self.num_antennas), 
            nn.Unflatten(1, (2 * self.num_antennas, 1))
        )

    def find_top_k_peaks(self, probs, K, neighborhood=5):
        """
        """
        B, C = probs.shape
        total_window = 2 * neighborhood + 1
    
        probs_unsqueezed = probs.unsqueeze(1) 
        pooled = F.max_pool1d(probs_unsqueezed, kernel_size=total_window, stride=1, padding=neighborhood)
        pooled = pooled.squeeze(1)
    
        is_peak = (probs == pooled)
        peak_vals = torch.where(is_peak, probs, torch.zeros_like(probs))
    
        topk_vals, topk_idxs = torch.topk(peak_vals, K, dim=1)
        topk_idxs, _ = topk_idxs.sort(dim=-1)
    
        return topk_idxs


    def capsule_output(self, scm, K, y=None):
        """
        """
        x = self.conv_layers(scm)
        x = self.prim_caps(x)
        x, r1 = self.capsule_1(x)
        length = x.norm(dim=-1)
        length = length / length.sum(dim=-1, keepdim=True)
        
        if y is None:
            index = self.find_top_k_peaks(length, K)
        else:
            index = torch.round(y).to(torch.int64) + 90

        N, L, D = x.shape
        indices = einops.repeat(index, "N K -> N K D", D=D)
        out = torch.gather(x, dim=1, index=indices)
            
        return out

    
    def forward(self, scm, K, y=None):
        """
        """
        x = self.conv_layers(scm) 
        x = self.prim_caps(x)
        x, r1 = self.capsule_1(x)
        length = x.norm(dim=-1)
        length = length / (length.sum(dim=-1, keepdim=True) + 1e-8)

        if y is None:
            index = self.find_top_k_peaks(length, K)
        else:
            index = torch.round(y).to(torch.int64) + 90

        W = torch.zeros(x.shape[0], 2 * self.num_antennas, K).to(x.device)

        for i in range(K):
            y = Variable(torch.zeros(length.size()).scatter_(1, index[:,i].view(-1, 1).cpu(), 1.).to(x.device))
            c = self.fc_layers((x * y[:, :, None]).view(x.size(0), -1))
            W[:, :, i] = self.pred_head(c).squeeze(-1)

        return W, length
    
class ULA:
    """
    A class implementing ULA.

    """
    def __init__(self, M):
        self.M = M

    def sind(self, degrees):
        """
        """
        return np.sin(np.deg2rad(degrees))

    def steering_vector(self, angle_deg):
        """
        """
        return np.exp(-1j*np.pi*np.arange(self.M).reshape(-1,1) * self.sind(angle_deg))
    

    def array_response(self, T, DOA_SOI, K, SNR, rng=None):
        """
        """
        if rng is None:
            rng = np.random.default_rng()
    
        gamma_SOI = 10**(SNR/10)
        
        A = self.steering_vector(DOA_SOI).astype('complex64') # M x K
        S = np.sqrt(gamma_SOI/2) * np.eye(K) @ (rng.standard_normal((K, T)).astype('float32') + 1j*(rng.standard_normal((K, T)).astype('float32'))) # K x T
        N = np.sqrt(1/2) * (rng.standard_normal((self.M, T)).astype('float32') + 1j*(rng.standard_normal((self.M, T))).astype('float32')) # M x T

        X = A @ S + N # M x T
        
        return X, S
    
    def compute_SCM(self, X):
        """
        """
        scm = (1/X.shape[1]) * X @ X.conj().T

        return scm
    
    def data_id(self):
        """
        """
        return "_M=" + str(self.M)
    
    def estimate_capon(self, X, K, DOA=None):
        """
        """
        SCM = self.compute_SCM(X)

        if DOA is None:
            Gr_size = 181
            dtheta= 180/(Gr_size-1)          
            angle_grid = np.arange(-90,90,dtheta)
            DOA, _ = self.compute_SCB(SCM, angle_grid, K)

        SCM_inv = solve(SCM, np.eye(SCM.shape[0]), assume_a='hermitian')
        a = self.steering_vector(DOA)

        if len(DOA) == 1:
            w = SCM_inv @ a / (a.conj().T @ SCM_inv @ a)
        else:
            #w = (SCM_inv @ a) @ solve(a.conj().T @ SCM_inv @ a, np.eye(K)) #np.linalg.inv(a.conj().T @ SCM_inv @ a)
            w = np.zeros((SCM.shape[0], K)).astype('complex64')
            for i in range(len(DOA)):
                a_i = self.steering_vector(DOA[i])
                w_i = SCM_inv @ a_i / (a_i.conj().T @ SCM_inv @ a_i)
                w[:,i] = w_i.reshape(-1)
            
        return w.conj().T @ X

    def estimate_mmse(self, X, K, qamma, DOA=None):
        SCM = self.compute_SCM(X)

        if DOA is None:
            Gr_size = 181
            dtheta= 180/(Gr_size-1)          
            angle_grid = np.arange(-90,90,dtheta)
            DOA, _ = self.compute_MUSIC(SCM, angle_grid, K)

        SCM_inv = solve(SCM, np.eye(SCM.shape[0]), assume_a='hermitian')
        a = self.steering_vector(DOA)

        w = SCM_inv @ a

        return qamma * np.eye(a.shape[1]) @ w.conj().T @ X
    
    def compute_SCB(self, SCM, angle_grid, K):
        """
        """
        A = self.steering_vector(angle_grid)
        SCM_inv = solve(SCM, np.eye(SCM.shape[0]), assume_a='hermitian')
        spatial_spectrum = 1/np.real(np.sum(A.conj().T @ SCM_inv *(A.T),axis=1))

        spatial_spectrum = 10*np.log10(np.real(spatial_spectrum))
        peak_inds, _ = find_peaks(spatial_spectrum) 
        DOA_inds = peak_inds[np.argpartition(spatial_spectrum[peak_inds], -K)[-K:]]
        DOAs = angle_grid[DOA_inds]

        if DOAs.size == 0:
            DOAs = np.array([0])

        return np.sort(DOAs), spatial_spectrum

    def compute_MUSIC(self, SCM, angle_grid, K):
        """
        """
        eval, evec = eig(SCM)
        idx = np.argpartition(eval, self.M-K)[:self.M-K]
        noise_evec = evec[:, idx]

        A = self.steering_vector(angle_grid)
        spatial_spectrum = 1/np.real(np.sum(A.conj().T @ noise_evec @ noise_evec.conj().T *(A.T),axis=1))

        spatial_spectrum = 10*np.log10(np.real(spatial_spectrum))
        peak_inds, _ = find_peaks(spatial_spectrum) 
        DOA_inds = peak_inds[np.argpartition(spatial_spectrum[peak_inds], -K)[-K:]]
        DOAs = angle_grid[DOA_inds]

        return np.sort(DOAs), spatial_spectrum


    def capon_ESE(self, DOA, qamma):
        """
        """
        A = self.steering_vector(DOA)

        if A.shape[1] == 1:
            return 1 / A.shape[0]
        else:
            se = np.zeros(A.shape[1])
            
            if qamma.shape[0] == 1:
                qamma = np.repeat(qamma, A.shape[1])
            
            for i in range(A.shape[1]):
                a = A[:,i]
                Q = np.delete(A, i, axis=1) @ np.diag(np.delete(qamma, i)) @ np.delete(A, i, axis=1).conj().T + np.eye(A.shape[0])
                iQ = np.linalg.solve(Q, np.eye(A.shape[0]))
                se[i] = 1 / np.real(a.conj().T @ iQ @ a)
                
            return se
            
    def mmse_ESE(self, DOA, qamma):
        """
        """
        qamma_cap = qamma + self.capon_ESE(DOA, qamma)
        se = (qamma / qamma_cap) * (qamma_cap - qamma)

        return se
    
    def estimate_dbf(self, X, K, model_dir, DOA=None):
        """
        """
        try:
            dbf = DeepBeamformer(self.M) #T
            id = self.data_id()
            dbf_path = os.path.abspath(model_dir)
            dbf.load_state_dict(torch.load(dbf_path + "/deepbeamformer" + id + ".pt", map_location=torch.device('cpu')))
        except FileNotFoundError:
            raise Exception("Trained deep beamformer doesn't exist")
        
        dbf.eval()
        SCM = self.compute_SCM(X)
        SCM = self.M * SCM / np.trace(SCM) 
        SCM = np.stack([SCM.real, SCM.imag, np.angle(SCM)], axis=0)
        SCM = torch.tensor(SCM, dtype=torch.float32).unsqueeze(0)
        
        if DOA is not None:
            DOA = torch.tensor(DOA, dtype=torch.float32).unsqueeze(0)
            
        W, DOA_dist = dbf(SCM, K, DOA)
        W = W.squeeze(0).detach().numpy()
        W = W[:self.M,:] + 1j * W[self.M:,:]

        DOA_est = dbf.find_top_k_peaks(DOA_dist, K).detach().numpy() - 90
        DOA_dist = DOA_dist.squeeze(0).detach().numpy()
        S = W.conj().T @ X
       
        return S, DOA_est, DOA_dist


    def dbf_caps_output(self, SCM, K, model_dir, DOA):
        """
        """
        try:
            dbf = DeepBeamformer(self.M) 
            id = self.data_id()
            dbf_path = os.path.abspath(model_dir)
            dbf.load_state_dict(torch.load(dbf_path + "/deepbeamformer" + id + ".pt", map_location=torch.device('cpu'))) #, map_location=torch.device('cpu')
        except FileNotFoundError:
            raise Exception("Trained deep beamformer doesn't exist")
        
        dbf.eval()
        out = dbf.capsule_output(SCM, K, DOA)

        return out
