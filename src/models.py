
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
#import tikzplotlib

class Capsule(nn.Module):
    """
    A class implementing capsule layers.

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

    def uniform_routing(self, u_hat):
        b = Variable(torch.zeros(u_hat.shape[0], self.out_caps_num, self.in_caps_num)).to(u_hat.device)
        c = F.softmax(b, dim=1)
        v = self.squash(torch.sum(c[:, :, :, None] * u_hat, dim=-2, keepdim=True))

        return v, c

    @torch.compile()   
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

class CapsFormer(nn.Module):
    """
    A class implementing CapsFormer.

    """
    def __init__(self, num_antennas): 
        super(CapsFormer, self).__init__()
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
            y = torch.zeros(length.size(), device=x.device).scatter_(1, index[:,i].view(-1, 1), 1.) 
            c = self.fc_layers((x * y[:, :, None]).view(x.size(0), -1))
            W[:, :, i] = self.pred_head(c).squeeze(-1)

        return W, length

class Encoder(nn.Module):
    """
    Encoder part of the INCM reconstruction network.
    """
    def __init__(self):
        super().__init__()
        self.conv64 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.mp64 = nn.MaxPool2d(kernel_size=2)

        self.conv128 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )

        self.mp128 = nn.MaxPool2d(kernel_size=2)

        self.conv256 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = self.conv64(x)
        skip1 = x
        x = self.mp64(x)
        x = self.conv128(x)
        skip2 = x
        x = self.mp128(x)
        x = self.conv256(x)

        return x, skip1, skip2


class Decoder(nn.Module):
    """
    Decoder part of the INCM reconstruction network.
    """
    def __init__(self):
        super().__init__()

        self.conv256 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.skipconv128 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )

        self.deconv256 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)

        self.conv128 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

        self.deconv128 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)

        self.skipconv64 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )

        self.conv64 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, skip1, skip2):
        x = self.conv256(x)

        skip2 = self.skipconv128(skip2)
        x = self.deconv256(x)
        x = torch.cat([skip2,  x], dim=1)

        x = self.conv128(x)

        skip1 = self.skipconv64(skip1)
        x = self.deconv128(x)
        x = torch.cat([skip1,  x], dim=1)

        x = self.conv64(x)
        
        return x

class ConjugateSymmetrization(nn.Module):
    """
    Conjugate symmetrization layer of the INCM reconstruction network.
    """
    def forward(self, Q):
        # Q: (B, 2, M, M)
        Q_real = Q[:, 0]           # (B, M, M)
        Q_imag = Q[:, 1]           # (B, M, M)

        Q_real_T = Q_real.transpose(-1, -2)
        Q_imag_T = Q_imag.transpose(-1, -2)

        V_real = 0.5 * (Q_real + Q_real_T)
        V_imag = 0.5 * (Q_imag - Q_imag_T)

        V = torch.stack([V_real, V_imag], dim=1)
        return V


class DeepReconstructionNet(nn.Module):
    """
    Deep INCM reconstruction network.
    """
    def __init__(self):
        super().__init__()

        self.enc = Encoder()
        self.dec = Decoder()
        self.conj_symm = ConjugateSymmetrization()

    def forward(self, x):
        x, skip1, skip2 = self.enc(x)
        x = self.dec(x, skip1, skip2)
        x = self.conj_symm(x)
        return x

class RNNBeamformer(nn.Module):
    """
    RNN based beamformer.
    """
    def __init__(self, M):
        super().__init__()

        self.gru = nn.GRU(input_size=1, hidden_size=512, num_layers=4, batch_first=True)
        self.fc = nn.Linear(512, 2*M)

    def forward(self, x):
        x, h = self.gru(x)
        x = self.fc(x[:, -1, :])
        return x
    
    
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
    
    def estimate_mvdr(self, X, K, DOA=None):
        """
        """
        SCM = self.compute_SCM(X)

        if DOA is None:
            Gr_size = 181
            dtheta= 180/(Gr_size-1)          
            angle_grid = np.arange(-90,90,dtheta)
            DOA, _ = self.compute_SCB(SCM, angle_grid, K)

        SCM_inv = solve(SCM, np.eye(SCM.shape[0])) #, assume_a='hermitian')
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
            
        return w #w.conj().T @ X

    def estimate_mmse(self, X, K, qamma, DOA=None):
        """
        """
        SCM = self.compute_SCM(X)

        if DOA is None:
            Gr_size = 181
            dtheta= 180/(Gr_size-1)          
            angle_grid = np.arange(-90,90,dtheta)
            DOA, _ = self.compute_MUSIC(SCM, angle_grid, K)

        SCM_inv = solve(SCM, np.eye(SCM.shape[0])) #, assume_a='hermitian')
        a = self.steering_vector(DOA)

        w = SCM_inv @ a
        
        return (qamma * np.eye(a.shape[1]) @ w.T).T #qamma * np.eye(a.shape[1]) @ w.conj().T @ X
    
    def mvdr_optimal_weights(self, qamma, DOA):
        """
        """
        w = np.zeros((self.M, len(DOA))).astype('complex64')
        A = self.steering_vector(DOA)

        for i in range(len(DOA)):
            a = self.steering_vector(DOA[i])
            Q = np.delete(A, i, axis=1) @ np.diag(np.delete(qamma, i)) @ np.delete(A, i, axis=1).conj().T + np.eye(A.shape[0])
            iQ = np.linalg.solve(Q, np.eye(A.shape[0]))
            w[:,i] = (iQ @ a / (a.conj().T @ iQ @ a)).reshape(-1)

        return w

    def mmse_optimal_weights(self, qamma, DOA):
        """
        """
        a = self.steering_vector(DOA)
        arr_cov = a @ np.diag(qamma) @ a.conj().T + np.eye(a.shape[0])
        arr_cov_inv = solve(arr_cov, np.eye(arr_cov.shape[0])) #, assume_a='hermitian')

        w = arr_cov_inv @ a

        return (qamma * np.eye(a.shape[1]) @ w.T).T
    
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


    def mvdr_ESE(self, DOA, qamma):
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
        qamma_cap = qamma + self.mvdr_ESE(DOA, qamma)
        se = (qamma / qamma_cap) * (qamma_cap - qamma)

        return se
    
    def estimate_dbf(self, X, K, model_dir, DOA=None):
        """
        """
        try:
            dbf = CapsFormer(self.M) #T
            #dbf = torch.compile(dbf)
            id = self.data_id()
            dbf_path = os.path.abspath(model_dir)
            dbf.load_state_dict(torch.load(dbf_path + "/capsformer" + id + ".pt", map_location=torch.device('cpu')))
        except FileNotFoundError:
            raise Exception("Trained CapsFormer doesn't exist")
        
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

        #S = W.conj().T @ X
       
        return W, DOA_est, DOA_dist


    def dbf_caps_output(self, SCM, K, model_dir, DOA):
        """
        """
        try:
            dbf = CapsFormer(self.M) 
            id = self.data_id()
            dbf_path = os.path.abspath(model_dir)
            dbf.load_state_dict(torch.load(dbf_path + "/capsformer" + id + ".pt", map_location=torch.device('cpu'))) #, map_location=torch.device('cpu')
        except FileNotFoundError:
            raise Exception("Trained CapsFormer doesn't exist")
        
        dbf.eval()
        out = dbf.capsule_output(SCM, K, DOA)

        return out
    
    def estimate_dirn(self, X, K, model_dir, DOA=None):
        """
        """
        try:
            dirn = DeepReconstructionNet()
            id = self.data_id()
            dirn_path = os.path.abspath(model_dir)
            dirn.load_state_dict(torch.load(dirn_path + "/deepincmreconstnet" + id + ".pt", map_location=torch.device('cpu')))
        except FileNotFoundError:
            raise Exception("Trained deep INCM reconstruction net doesn't exist")
        
        dirn.eval()
        SCM = self.compute_SCM(X)

        if DOA is None:
            Gr_size = 181
            dtheta= 180/(Gr_size-1)          
            angle_grid = np.arange(-90,90,dtheta)
            DOA, _ = self.compute_SCB(SCM, angle_grid, K)

        #SCM = self.M * SCM / np.trace(SCM)

        if K == 1:
            SCM = np.stack([SCM.real, SCM.imag, np.angle(SCM), np.full(SCM.shape, DOA)], axis=0)
            SCM = torch.tensor(SCM, dtype=torch.float32).unsqueeze(0)
            INCM_est = dirn(SCM)
            INCM_est = INCM_est.squeeze(0).detach().numpy()
            INCM_est = INCM_est[0] + 1j * INCM_est[1]

            a = self.steering_vector(DOA)
            w = INCM_est @ a / (a.conj().T @ INCM_est @ a)
        else:
            w = np.zeros((SCM.shape[0], K)).astype('complex64')
            for i in range(K):
                SCM_i = np.stack([SCM.real, SCM.imag, np.angle(SCM), np.full(SCM.shape, DOA[i])], axis=0)
                SCM_i = torch.tensor(SCM_i, dtype=torch.float32).unsqueeze(0)
                INCM_est = dirn(SCM_i)
                INCM_est = INCM_est.squeeze(0).detach().numpy()
                INCM_est = INCM_est[0] + 1j * INCM_est[1]

                a = self.steering_vector(DOA[i])
                w[:,i] = (INCM_est @ a / (a.conj().T @ INCM_est @ a)).reshape(-1)
       
        return w
    
    def estimate_rnnbf(self, X, K, model_dir, DOA=None):
        """
        """
        try:
            rnnbf = RNNBeamformer(self.M)
            id = self.data_id()
            rnnbf_path = os.path.abspath(model_dir)
            rnnbf.load_state_dict(torch.load(rnnbf_path + "/rnnbeamformer" + id + ".pt", map_location=torch.device('cpu')))
        except FileNotFoundError:
            raise Exception("Trained RNN beamformer doesn't exist")
        
        rnnbf.eval()
        SCM = self.compute_SCM(X)

        if DOA is None:
            Gr_size = 181
            dtheta= 180/(Gr_size-1)          
            angle_grid = np.arange(-90,90,dtheta)
            DOA, _ = self.compute_MUSIC(SCM, angle_grid, K)
        
        DOA = (torch.tensor(DOA, dtype=torch.float32).unsqueeze(0).unsqueeze(-1) + 90) / 180

        if K == 1:
            w = rnnbf(DOA)
            w = w.squeeze(0).detach().numpy()
            w = (w[:self.M] + 1j * w[self.M:]).reshape(-1,1)
        else:
            w = np.zeros((SCM.shape[0], K)).astype('complex64')
            for i in range(K):
                w_i = rnnbf(torch.roll(DOA, i, 1))
                w_i = w_i.squeeze(0).detach().numpy()
                w_i = w_i[:self.M] + 1j * w_i[self.M:]
                w[:,i] = w_i.reshape(-1)
       
        return w
