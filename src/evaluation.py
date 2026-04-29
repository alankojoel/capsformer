
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from scipy.linalg import solve
from src.models import ULA
from src.utils import log_stats
import tikzplotlib

class Tester:
    """
    A class for evaluating the performance of the CapsFormer.

    """
    def __init__(self, **kwargs):
        self.M = kwargs["M"]
        self.T = kwargs["T"]
        self.K = kwargs["K"]
        self.DOA_SOI_start = kwargs["DOA_SOI_range"][0]
        self.DOA_SOI_end = kwargs["DOA_SOI_range"][1]
        self.SNR_start = kwargs["SNR_range"][0]
        self.SNR_end = kwargs["SNR_range"][1]
        self.SNR_spacing = kwargs["SNR_spacing"]
        self.random_state = kwargs["test_random_state"]
        self.MC_trials = kwargs["MC_trials"]
        self.model_dir = kwargs["model_dir"]
        self.test_dir = kwargs["test_dir"]

    
    def estimate_SOI(self, arr, DOA, K, DOA_SOI_true, DOA_SOI_est, SNR, rng=None):
        """
        """
        W_est = np.empty((5, len(SNR)), dtype=object)
        SOI_est = np.empty((5, len(SNR)), dtype=object)
        SOI_true = np.empty(len(SNR), dtype=object)

        for i, SNR_i in enumerate(SNR):
            X, S = arr.array_response(self.T, DOA, len(DOA), SNR_i, rng=rng)
            if DOA_SOI_true is not None:
                SOI_true[i] = S[np.where(np.isin(DOA, DOA_SOI_true))[0]]
                SNR_i = SNR_i[np.where(np.isin(DOA, DOA_SOI_true))[0]]
            else:
                SOI_true[i] = S
            w_mvdr = arr.estimate_mvdr(X, K, DOA_SOI_est)
            w_mmse = arr.estimate_mmse(X, K, 10**(SNR_i/10), DOA_SOI_est)
            w_dirn = arr.estimate_dirn(X, K, self.model_dir, DOA_SOI_est)
            if DOA_SOI_true is not None:
                w_rnnbf = arr.estimate_rnnbf(X, K, self.model_dir, np.insert(np.delete(DOA, np.where(np.isin(DOA, DOA_SOI_true))[0]), 0, DOA_SOI_est))
            else:
                w_rnnbf = arr.estimate_rnnbf(X, K, self.model_dir, DOA_SOI_est)
            w_dbf, _, _ = arr.estimate_dbf(X, K, self.model_dir, DOA_SOI_est)
            W_est[0][i] = w_mvdr
            W_est[1][i] = w_mmse
            W_est[2][i] = w_dirn
            W_est[3][i] = w_rnnbf
            W_est[4][i] = w_dbf
            SOI_est[0][i] = w_mvdr.conj().T @ X
            SOI_est[1][i] = w_mmse.conj().T @ X
            SOI_est[2][i] = w_dirn.conj().T @ X
            SOI_est[3][i] = w_rnnbf.conj().T @ X
            SOI_est[4][i] = w_dbf.conj().T @ X
        
        return W_est, SOI_est, SOI_true
    
    def estimate_single_sample(self, arr, DOAs, DOAs_SOI, DOAs_SOI_perturb, SNRs, rng=None):
        """
        """
        if rng is None:
            rng = np.random.default_rng(self.random_state)

        DOAs_SOI_true = []
        DOAs_SOI_est = []
        Ws_est = []
        SOIs_est = []
        SOIs_true = []

        for i in range(len(DOAs)):
            DOA_i = np.array(DOAs[i])
            if DOAs_SOI is not None:
                DOA_SOI_i_true = np.array(DOAs_SOI[i])
                K = len(DOA_SOI_i_true)
                if DOAs_SOI_perturb:
                    DOA_SOI_i_est = np.clip(DOA_SOI_i_true + rng.uniform(-2.5, 2.5, K), -90, 90)
                else:
                    DOA_SOI_i_est = DOA_SOI_i_true
            else:
                DOA_SOI_i_true = None
                DOA_SOI_i_est = None
                K = len(DOA_i)
            W_i_est, SOI_i_est, SOI_i_true = self.estimate_SOI(arr, DOA_i, K, DOA_SOI_i_true, DOA_SOI_i_est, SNRs, rng=rng)
            DOAs_SOI_true.append(DOA_SOI_i_true)
            DOAs_SOI_est.append(DOA_SOI_i_est)
            Ws_est.append(W_i_est)
            SOIs_est.append(SOI_i_est)
            SOIs_true.append(SOI_i_true)

        DOAs_SOI_true = np.array(DOAs_SOI_true)
        DOAs_SOI_est = np.array(DOAs_SOI_est)
        Ws_est = np.array(Ws_est)
        SOIs_est = np.array(SOIs_est)
        SOIs_true = np.array(SOIs_true)

        SINR = self.compute_SINR(arr, SNRs, Ws_est, DOAs, DOAs_SOI_true)
        NMSE = self.compute_NMSE(SNRs, SOIs_true, SOIs_est)
        BIAS = self.compute_BIAS(SNRs, SOIs_true, SOIs_est)

        return SINR, NMSE, BIAS

    def compute_SINR(self, arr, SNRs, Ws_est, DOAs, DOAs_SOI_true):
        """
        """
        SINR =  [[] for _ in range(6)]
        
        num_DOA = Ws_est.shape[0] if Ws_est[0][0][0].shape[1] == 1 else Ws_est[0][0][0].shape[1]

        for i in range(6): 
            SINR[i]= np.zeros((num_DOA, len(SNRs)))

        for i in range(len(DOAs)):
            DOA_i = DOAs[i]
            DOA_SOI_i_true = DOAs_SOI_true[i]
            A = arr.steering_vector(DOA_i)
            for j in range(len(SNRs)):
                SNR_j = SNRs[j]
                qamma_j = 10**(SNR_j/10)
                if DOA_SOI_i_true is not None:
                    for k in range(len(DOA_SOI_i_true)):
                        l = np.where(np.isin(DOA_i, DOA_SOI_i_true[k]))[0]
                        a = arr.steering_vector(DOA_SOI_i_true[k])
                        Q = np.delete(A, l, axis=1) @ np.diag(np.delete(qamma_j, l)) @ np.delete(A, l, axis=1).conj().T + np.eye(A.shape[0])
                        iQ = np.linalg.solve(Q, np.eye(A.shape[0]))
                        w_opt = iQ @ a / (a.conj().T @ iQ @ a)
                        SINR[0][i+k][j] = ((qamma_j[l] * np.abs(Ws_est[i+k][0][j].conj().T @ a)**2) / np.real(Ws_est[i+k][0][j].conj().T @ Q @ Ws_est[i+k][0][j])).item()
                        SINR[1][i+k][j] = ((qamma_j[l] * np.abs(Ws_est[i+k][1][j].conj().T @ a)**2) / np.real(Ws_est[i+k][1][j].conj().T @ Q @ Ws_est[i+k][1][j])).item()
                        SINR[2][i+k][j] = ((qamma_j[l] * np.abs(Ws_est[i+k][2][j].conj().T @ a)**2) / np.real(Ws_est[i+k][2][j].conj().T @ Q @ Ws_est[i+k][2][j])).item()
                        SINR[3][i+k][j] = ((qamma_j[l] * np.abs(Ws_est[i+k][3][j].conj().T @ a)**2) / np.real(Ws_est[i+k][3][j].conj().T @ Q @ Ws_est[i+k][3][j])).item()
                        SINR[4][i+k][j] = ((qamma_j[l] * np.abs(Ws_est[i+k][4][j].conj().T @ a)**2) / np.real(Ws_est[i+k][4][j].conj().T @ Q @ Ws_est[i+k][4][j])).item()
                        SINR[5][i+k][j] = ((qamma_j[l] * np.abs(w_opt.conj().T @ a)**2) / np.real(w_opt.conj().T @ Q @ w_opt)).item()
                else:
                    for k in range(len(DOA_i)):
                        l = np.where(np.isin(DOA_i, DOA_i[k]))[0]
                        a = arr.steering_vector(DOA_i[k])
                        Q = np.delete(A, l, axis=1) @ np.diag(np.delete(qamma_j, l)) @ np.delete(A, l, axis=1).conj().T + np.eye(A.shape[0])
                        iQ = np.linalg.solve(Q, np.eye(A.shape[0]))
                        w_opt = iQ @ a / (a.conj().T @ iQ @ a)
                        SINR[0][i+k][j] = ((qamma_j[l] * np.abs(Ws_est[i][0][j][:,k].conj().T @ a)**2) / np.real(Ws_est[i][0][j][:,k].conj().T @ Q @ Ws_est[i][0][j][:,k])).item()
                        SINR[1][i+k][j] = ((qamma_j[l] * np.abs(Ws_est[i][1][j][:,k].conj().T @ a)**2) / np.real(Ws_est[i][1][j][:,k].conj().T @ Q @ Ws_est[i][1][j][:,k])).item()
                        SINR[2][i+k][j] = ((qamma_j[l] * np.abs(Ws_est[i][2][j][:,k].conj().T @ a)**2) / np.real(Ws_est[i][2][j][:,k].conj().T @ Q @ Ws_est[i][2][j][:,k])).item()
                        SINR[3][i+k][j] = ((qamma_j[l] * np.abs(Ws_est[i][3][j][:,k].conj().T @ a)**2) / np.real(Ws_est[i][3][j][:,k].conj().T @ Q @ Ws_est[i][3][j][:,k])).item()
                        SINR[4][i+k][j] = ((qamma_j[l] * np.abs(Ws_est[i][4][j][:,k].conj().T @ a)**2) / np.real(Ws_est[i][4][j][:,k].conj().T @ Q @ Ws_est[i][4][j][:,k])).item()
                        SINR[5][i+k][j] = ((qamma_j[l] * np.abs(w_opt.conj().T @ a)**2) / np.real(w_opt.conj().T @ Q @ w_opt)).item()

        return np.array(SINR)
    
    def compute_NMSE(self, SNRs, SOIs_true, SOIs_est):
        """
        """
        NMSE =  [[] for _ in range(5)]
        
        num_DOA = SOIs_true.shape[0] if SOIs_true[0][0].shape[0] == 1 else SOIs_true[0][0].shape[0]
        
        if SOIs_true.shape[0] > 1:
            assert SOIs_true[0][0].shape[0] == 1
            
        for i in range(5): 
            NMSE[i]= np.zeros((num_DOA, len(SNRs)))

        for i in range(SOIs_true.shape[0]):
            for j in range(len(SNRs)):
                SOI_true = SOIs_true[i][j]
                if SOIs_true[0][0].shape[0] == 1:
                    NMSE[0][i][j] = (np.mean(np.abs(SOIs_est[i][0][j] - SOI_true)**2, axis=1) / np.mean(np.abs(SOI_true)**2, axis=1)).item()
                    NMSE[1][i][j] = (np.mean(np.abs(SOIs_est[i][1][j] - SOI_true)**2, axis=1) / np.mean(np.abs(SOI_true)**2, axis=1)).item()
                    NMSE[2][i][j] = (np.mean(np.abs(SOIs_est[i][2][j] - SOI_true)**2, axis=1) / np.mean(np.abs(SOI_true)**2, axis=1)).item()
                    NMSE[3][i][j] = (np.mean(np.abs(SOIs_est[i][3][j] - SOI_true)**2, axis=1) / np.mean(np.abs(SOI_true)**2, axis=1)).item()
                    NMSE[4][i][j] = (np.mean(np.abs(SOIs_est[i][4][j] - SOI_true)**2, axis=1) / np.mean(np.abs(SOI_true)**2, axis=1)).item()
                else:
                    NMSE[0][:,j] = np.mean(np.abs(SOIs_est[i][0][j] - SOI_true)**2, axis=1) / np.mean(np.abs(SOI_true)**2, axis=1)
                    NMSE[1][:,j] = np.mean(np.abs(SOIs_est[i][1][j] - SOI_true)**2, axis=1) / np.mean(np.abs(SOI_true)**2, axis=1)
                    NMSE[2][:,j] = np.mean(np.abs(SOIs_est[i][2][j] - SOI_true)**2, axis=1) / np.mean(np.abs(SOI_true)**2, axis=1)
                    NMSE[3][:,j] = np.mean(np.abs(SOIs_est[i][3][j] - SOI_true)**2, axis=1) / np.mean(np.abs(SOI_true)**2, axis=1)
                    NMSE[4][:,j] = np.mean(np.abs(SOIs_est[i][4][j] - SOI_true)**2, axis=1) / np.mean(np.abs(SOI_true)**2, axis=1)

        return np.array(NMSE)

    def compute_ESE(self, arr, DOAs, DOAs_SOI, SNRs):
        """
        """
        ESE =  [[] for _ in range(2)]

        num_DOA = DOAs.shape[0] if DOAs.shape[1] == 1 else DOAs.shape[1]
        
        if DOAs.shape[0] > 1:
            assert DOAs.shape[1] == 1

        for i in range(2):
            ESE[i] = np.zeros((num_DOA, len(SNRs)))

        for i in range(DOAs.shape[0]):
            DOA_i = np.array(DOAs[i])
            for j in range(len(SNRs)):
                qamma_j = 10**(SNRs[j]/10)
                if DOAs.shape[1] == 1:
                    ESE[0][i][j] = arr.mvdr_ESE(DOA_i, qamma_j) / qamma_j
                    ESE[1][i][j] = arr.mmse_ESE(DOA_i, qamma_j) / qamma_j
                else:
                    ESE[0][:,j] = arr.mvdr_ESE(DOA_i, qamma_j) / qamma_j
                    ESE[1][:,j] = arr.mmse_ESE(DOA_i, qamma_j) / qamma_j

        ESE = np.array(ESE)

        if DOAs_SOI is not None:
            ESE = ESE[:, np.where(np.isin(DOAs.flatten(), DOAs_SOI.flatten()))[0],:]
                    
        return ESE
        

    def compute_BIAS(self, SNRs, SOIs_true, SOIs_est):
        """
        """
        BIAS =  [[] for _ in range(5)]

        num_DOA = SOIs_true.shape[0] if  SOIs_true[0][0].shape[0] == 1 else SOIs_true[0][0].shape[0]
        if SOIs_true.shape[0] > 1:
            assert SOIs_true[0][0].shape[0] == 1
            
        for i in range(5): 
            BIAS[i]= np.zeros((num_DOA, len(SNRs)))

        for i in range(SOIs_true.shape[0]):
            for j in range(len(SNRs)):
                SOI_true = SOIs_true[i][j]
                if SOIs_true[0][0].shape[0] == 1:
                    BIAS[0][i][j] = ((np.mean(np.abs(SOIs_est[i][0][j])**2, axis=1) - np.mean(np.abs(SOI_true)**2, axis=1)) / np.mean(np.abs(SOI_true)**2, axis=1)).item()
                    BIAS[1][i][j] = ((np.mean(np.abs(SOIs_est[i][1][j])**2, axis=1) - np.mean(np.abs(SOI_true)**2, axis=1)) / np.mean(np.abs(SOI_true)**2, axis=1)).item()
                    BIAS[2][i][j] = ((np.mean(np.abs(SOIs_est[i][2][j])**2, axis=1) - np.mean(np.abs(SOI_true)**2, axis=1)) / np.mean(np.abs(SOI_true)**2, axis=1)).item()
                    BIAS[3][i][j] = ((np.mean(np.abs(SOIs_est[i][3][j])**2, axis=1) - np.mean(np.abs(SOI_true)**2, axis=1)) / np.mean(np.abs(SOI_true)**2, axis=1)).item()
                    BIAS[4][i][j] = ((np.mean(np.abs(SOIs_est[i][4][j])**2, axis=1) - np.mean(np.abs(SOI_true)**2, axis=1)) / np.mean(np.abs(SOI_true)**2, axis=1)).item()
                else:
                    BIAS[0][:,j] = (np.mean(np.abs(SOIs_est[i][0][j])**2, axis=1) - np.mean(np.abs(SOI_true)**2, axis=1)) / np.mean(np.abs(SOI_true)**2, axis=1)
                    BIAS[1][:,j] = (np.mean(np.abs(SOIs_est[i][1][j])**2, axis=1) - np.mean(np.abs(SOI_true)**2, axis=1)) / np.mean(np.abs(SOI_true)**2, axis=1)
                    BIAS[2][:,j] = (np.mean(np.abs(SOIs_est[i][2][j])**2, axis=1) - np.mean(np.abs(SOI_true)**2, axis=1)) / np.mean(np.abs(SOI_true)**2, axis=1)
                    BIAS[3][:,j] = (np.mean(np.abs(SOIs_est[i][3][j])**2, axis=1) - np.mean(np.abs(SOI_true)**2, axis=1)) / np.mean(np.abs(SOI_true)**2, axis=1)
                    BIAS[4][:,j] = (np.mean(np.abs(SOIs_est[i][4][j])**2, axis=1) - np.mean(np.abs(SOI_true)**2, axis=1)) / np.mean(np.abs(SOI_true)**2, axis=1)

        return np.array(BIAS)
    
    def estimate_MC_trials(self, mode, DOAs, DOAs_SOI, DOAs_SOI_perturb, SNRs, MC_trials=None):
        """
        """
        if MC_trials is None:
            MC_trials = self.MC_trials
        
        arr = ULA(self.M)
        rng = np.random.default_rng(self.random_state)

        if mode == "s":
            if DOAs_SOI is not None:
                assert np.all(DOAs == DOAs_SOI)
                DOAs_SOI = DOAs_SOI.reshape(-1,1)
            DOAs = DOAs.reshape(-1,1)
            SINR = np.zeros((6, DOAs.shape[0], len(SNRs)))
            NMSE = np.zeros((5, DOAs.shape[0], len(SNRs)))
            BIAS = np.zeros((5, DOAs.shape[0], len(SNRs)))
        elif mode == "m":
            if DOAs_SOI is not None:
                assert np.all(np.isin(DOAs_SOI, DOAs))
                K = len(DOAs_SOI)
                DOAs_SOI = DOAs_SOI.reshape(1,-1)
            else:
                K = len(DOAs)
            DOAs = DOAs.reshape(1,-1) 
            SINR = np.zeros((6, K, len(SNRs)))      
            NMSE = np.zeros((5, K, len(SNRs)))
            BIAS = np.zeros((5, K, len(SNRs)))
        else:
            raise Exception("Invalid mode.")
            
        pbar = tqdm(total=MC_trials, position=0, leave=True)

        for i in range(MC_trials):
            SINR_i, NMSE_i, BIAS_i = self.estimate_single_sample(arr, DOAs, DOAs_SOI, DOAs_SOI_perturb, SNRs, rng=rng)
            SINR += SINR_i
            NMSE += NMSE_i
            BIAS += BIAS_i
            curr_NMSE_mean = np.mean(NMSE / (i + 1), axis=(1,2))
            curr_SINR_mean = np.mean(SINR / (i + 1), axis=(1,2))
            curr_BIAS_mean = np.mean(BIAS / (i + 1), axis=(1,2))
            DOA_s = ",".join(f"{DOAs[i]}" for i in range(DOAs.shape[0]))
            DOA_SOI_s = ",".join(f"{DOAs_SOI[i]}" for i in range(DOAs_SOI.shape[0])) if DOAs_SOI is not None else DOA_s
            SNR_s = ",".join(f"{SNRs[i]}" for i in range(SNRs.shape[0]))
            desc = (f"DOA: {DOA_s}, DOA_SOI: {DOA_SOI_s}, Apply perturbations: {DOAs_SOI_perturb}, MC trials: {i + 1}, Average NMSE: MVDR Beamformer: {curr_NMSE_mean[0]:.4f}, MMSE Beamformer: {curr_NMSE_mean[1]:.4f}, Deep INCM Reconst Net: {curr_NMSE_mean[2]:.4f}, GRU Beamformer: {curr_NMSE_mean[3]:.4f}, CapsFormer: {curr_NMSE_mean[4]:.4f}" 
                    f"Average SINR: MVDR Beamformer: {curr_SINR_mean[0]:.4f}, MMSE Beamformer: {curr_SINR_mean[1]:.4f}, Deep INCM Reconst Net: {curr_SINR_mean[2]:.4f}, GRU Beamformer: {curr_SINR_mean[3]:.4f}, CapsFormer: {curr_SINR_mean[4]:.4f}, Optimal: {curr_SINR_mean[5]:.4f}")
            pbar.set_description(desc)  
            pbar.update()

        pbar.close()

        ESE = self.compute_ESE(arr, DOAs, DOAs_SOI, SNRs)

        return SINR / MC_trials, np.mean(SINR / MC_trials, axis=2), NMSE / MC_trials, np.mean(NMSE / MC_trials, axis=2), ESE, np.mean(ESE, axis=2), BIAS / MC_trials, np.mean(BIAS / MC_trials, axis=2)
    
    def compare_metrics(self, mode, DOAs, DOAs_SOI, DOAs_SOI_perturb, SNRs, MC_trials=None):
        """ 
        """
        if MC_trials is None:
            MC_trials = self.MC_trials
        
        SINR, SINR_DOA, NMSE, NMSE_DOA, ESE, ESE_DOA, BIAS, BIAS_DOA = self.estimate_MC_trials(mode, DOAs, DOAs_SOI, DOAs_SOI_perturb, SNRs, MC_trials=MC_trials)
        
        test_path = os.path.abspath(self.test_dir)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        log_path = os.path.join(test_path, "test.log")
        if os.path.exists(log_path):
            os.remove(log_path)
        
        with open(log_path, 'w') as fp:
            pass
        
        if mode == "s":
            SNRs = np.stack([SNRs.reshape(-1) for _ in range(DOAs.shape[0])], axis=1)

        log_stats(log_path, f"DOAs: {', '.join(f'{x}' for x in DOAs)}")

        if DOAs_SOI is not None:
            log_stats(log_path, f"DOAs_SOI: {', '.join(f'{x}' for x in DOAs_SOI)}")
            DOAs_act = DOAs_SOI
            SNRs_act = SNRs[:, np.where(np.isin(DOAs, DOAs_SOI))[0]]
        else:
            DOAs_act = DOAs
            SNRs_act = SNRs
        
        for i in range(DOAs.shape[0]):
            log_stats(log_path, f"SNRs DOA={DOAs[i]}: {SNRs[:,i]}")
            
        log_stats(log_path, f"Monte Carlo trials: {MC_trials}")
        
        log_stats(log_path, f"(SINR DOAs) MVDR Beamformer: {', '.join(f'{x:.4f}' for x in SINR_DOA[0])}")
        log_stats(log_path, f"(SINR DOAs) MMSE Beamformer: {', '.join(f'{x:.4f}' for x in SINR_DOA[1])}")
        log_stats(log_path, f"(SINR DOAs) Deep INCM Reconst Net: {', '.join(f'{x:.4f}' for x in SINR_DOA[2])}")
        log_stats(log_path, f"(SINR DOAs) GRU Beamformer: {', '.join(f'{x:.4f}' for x in SINR_DOA[3])}")
        log_stats(log_path, f"(SINR DOAs) CapsFormer: {', '.join(f'{x:.4f}' for x in SINR_DOA[4])}")
        log_stats(log_path, f"(SINR DOAs) Optimal: {', '.join(f'{x:.4f}' for x in SINR_DOA[5])}")

        log_stats(log_path, f"(NMSE DOAs) MVDR Beamformer: {', '.join(f'{x:.4f}' for x in NMSE_DOA[0])}")
        log_stats(log_path, f"(NMSE DOAs) MMSE Beamformer: {', '.join(f'{x:.4f}' for x in NMSE_DOA[1])}")
        log_stats(log_path, f"(NMSE DOAs) Deep INCM Reconst Net: {', '.join(f'{x:.4f}' for x in NMSE_DOA[2])}")
        log_stats(log_path, f"(NMSE DOAs) GRU Beamformer: {', '.join(f'{x:.4f}' for x in NMSE_DOA[3])}")
        log_stats(log_path, f"(NMSE DOAs) CapsFormer: {', '.join(f'{x:.4f}' for x in NMSE_DOA[4])}")
        log_stats(log_path, f"(NMSE DOAs) Optimal: {', '.join(f'{x:.4f}' for x in ESE_DOA[1])}")

        for i in range(DOAs_act.shape[0]):
            log_stats(log_path, f"(SINR SNRs DOA={DOAs_act[i]}) MVDR Beamformer: {', '.join(f'{x:.4f}' for x in SINR[0][i])}")
            log_stats(log_path, f"(SINR SNRs DOA={DOAs_act[i]}) MMSE Beamformer: {', '.join(f'{x:.4f}' for x in SINR[1][i])}")
            log_stats(log_path, f"(SINR SNRs DOA={DOAs_act[i]}) Deep INCM Reconst Net: {', '.join(f'{x:.4f}' for x in SINR[2][i])}")
            log_stats(log_path, f"(SINR SNRs DOA={DOAs_act[i]}) GRU Beamformer: {', '.join(f'{x:.4f}' for x in SINR[3][i])}")
            log_stats(log_path, f"(SINR SNRs DOA={DOAs_act[i]}) CapsFormer: {', '.join(f'{x:.4f}' for x in SINR[4][i])}")
            log_stats(log_path, f"(SINR SNRs DOA={DOAs_act[i]}) Optimal: {', '.join(f'{x:.4f}' for x in SINR[5][i])}")

        for i in range(DOAs_act.shape[0]):
            log_stats(log_path, f"(NMSE SNRs DOA={DOAs_act[i]}) MVDR Beamformer: {', '.join(f'{x:.4f}' for x in NMSE[0][i])}")
            log_stats(log_path, f"(NMSE SNRs DOA={DOAs_act[i]}) MMSE Beamformer: {', '.join(f'{x:.4f}' for x in NMSE[1][i])}")
            log_stats(log_path, f"(NMSE SNRs DOA={DOAs_act[i]}) Deep INCM Reconst Net: {', '.join(f'{x:.4f}' for x in NMSE[2][i])}")
            log_stats(log_path, f"(NMSE SNRs DOA={DOAs_act[i]}) GRU Beamformer: {', '.join(f'{x:.4f}' for x in NMSE[3][i])}")
            log_stats(log_path, f"(NMSE SNRs DOA={DOAs_act[i]}) CapsFormer: {', '.join(f'{x:.4f}' for x in NMSE[4][i])}")
            log_stats(log_path, f"(NMSE SNRs DOA={DOAs_act[i]}) Optimal: {', '.join(f'{x:.4f}' for x in ESE[1][i])}")

        log_stats(log_path, f"(BIAS DOAs) MVDR Beamformer: {', '.join(f'{x:.4f}' for x in BIAS_DOA[0])}")
        log_stats(log_path, f"(BIAS DOAs) MMSE Beamformer: {', '.join(f'{x:.4f}' for x in BIAS_DOA[1])}")
        log_stats(log_path, f"(BIAS DOAs) Deep INCM Reconst Net: {', '.join(f'{x:.4f}' for x in BIAS_DOA[2])}")
        log_stats(log_path, f"(BIAS DOAs) GRU Beamformer: {', '.join(f'{x:.4f}' for x in BIAS_DOA[3])}")
        log_stats(log_path, f"(BIAS DOAs) CapsFormer: {', '.join(f'{x:.4f}' for x in BIAS_DOA[4])}")

        for i in range(DOAs_act.shape[0]):
            log_stats(log_path, f"(BIAS SNRs DOA={DOAs_act[i]}) MVDR Beamformer: {', '.join(f'{x:.4f}' for x in BIAS[0][i])}")
            log_stats(log_path, f"(BIAS SNRs DOA={DOAs_act[i]}) MMSE Beamformer: {', '.join(f'{x:.4f}' for x in BIAS[1][i])}")
            log_stats(log_path, f"(BIAS SNRs DOA={DOAs_act[i]}) Deep INCM Reconst Net: {', '.join(f'{x:.4f}' for x in BIAS[2][i])}")
            log_stats(log_path, f"(BIAS SNRs DOA={DOAs_act[i]}) GRU Beamformer: {', '.join(f'{x:.4f}' for x in BIAS[3][i])}")
            log_stats(log_path, f"(BIAS SNRs DOA={DOAs_act[i]}) CapsFormer: {', '.join(f'{x:.4f}' for x in BIAS[4][i])}")

        self.plot_SINR(SINR_DOA, SINR, DOAs_act, DOAs_SOI_perturb, SNRs_act, test_path)

        self.plot_NMSE(NMSE_DOA, NMSE, ESE, ESE_DOA, DOAs_act, DOAs_SOI_perturb, SNRs_act, test_path)

        self.plot_BIAS(BIAS_DOA, BIAS, DOAs_act, DOAs_SOI_perturb, SNRs_act, test_path)
    
    def plot_SINR(self, SINR_DOA, SINR, DOAs, DOAs_SOI_perturb, SNRs, test_path):
        """
        """
        fig, ax = plt.subplots(len(DOAs) + 1, 1, figsize=(6, (len(DOAs) + 1) * 6))

        for i in range(len(DOAs) + 1):
            if i == 0:
                ax[i].plot(DOAs, 10*np.log10(SINR_DOA[0]), marker="D", linestyle="none", mfc="none", label="MVDR Beamformer")
                ax[i].plot(DOAs, 10*np.log10(SINR_DOA[1]), marker="o", linestyle="none", mfc="none", label="MMSE Beamformer")
                ax[i].plot(DOAs, 10*np.log10(SINR_DOA[2]), marker="s", linestyle="none", mfc="none", label="Deep INCM Reconst Net")
                ax[i].plot(DOAs, 10*np.log10(SINR_DOA[3]), marker=(5,2), linestyle="none", mfc="none", label="GRU Beamformer")
                ax[i].plot(DOAs, 10*np.log10(SINR_DOA[4]), marker="x", linestyle="none", mfc="none", label="CapsFormer")
                ax[i].plot(DOAs, 10*np.log10(SINR_DOA[5]), marker="+", linestyle="none", mfc="none", label="Optimal")
                ax[i].set_title('SINR DOAs')
                ax[i].set_ylabel('SINR')
                ax[i].set_xlabel('DOA')
                #ax[i].legend()
                ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)
            else:
                ax[i].plot(SNRs[:,i-1], 10*np.log10(SINR[0][i-1]), marker="D", linestyle="none", mfc="none", label="MVDR Beamformer")
                ax[i].plot(SNRs[:,i-1], 10*np.log10(SINR[1][i-1]), marker="o", linestyle="none", mfc="none", label="MMSE Beamformer")
                ax[i].plot(SNRs[:,i-1], 10*np.log10(SINR[2][i-1]), marker="s", linestyle="none", mfc="none", label="Deep INCM Reconst Net")
                ax[i].plot(SNRs[:,i-1], 10*np.log10(SINR[3][i-1]), marker=(5,2), linestyle="none", mfc="none", label="GRU Beamformer")
                ax[i].plot(SNRs[:,i-1], 10*np.log10(SINR[4][i-1]), marker="x", linestyle="none", mfc="none", label="CapsFormer")
                ax[i].plot(SNRs[:,i-1], 10*np.log10(SINR[5][i-1]), marker="+", linestyle="none", mfc="none", label="Optimal")
                ax[i].set_title(f'SINR DOA={DOAs[i-1]}')
                ax[i].set_ylabel('SINR')
                ax[i].set_xlabel('SNR')
                #ax[i].legend()
                ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)

        tikzplotlib.save(test_path + "/" + "DOA=" + str(DOAs) + "_DOA_SOI_perturb=" + str(DOAs_SOI_perturb) + "_test_SINR_plot.tex")
        fig.savefig(test_path + "/" + "DOA=" + str(DOAs)+ "_DOA_SOI_perturb=" + str(DOAs_SOI_perturb) + "_test_SINR_plot.pdf", bbox_inches="tight")
    
    def plot_NMSE(self, NMSE_DOA, NMSE, ESE, ESE_DOA, DOAs, DOAs_SOI_perturb, SNRs, test_path):
        """
        """
        fig, ax = plt.subplots(len(DOAs) + 1, 1, figsize=(6, (len(DOAs) + 1) * 6))

        for i in range(len(DOAs) + 1):
            if i == 0:
                ax[i].plot(DOAs, NMSE_DOA[0], marker="D", linestyle="none", mfc="none", label="MVDR Beamformer")
                ax[i].plot(DOAs, NMSE_DOA[1], marker="o", linestyle="none", mfc="none", label="MMSE Beamformer")
                ax[i].plot(DOAs, NMSE_DOA[2], marker="s", linestyle="none", mfc="none", label="Deep INCM Reconst Net")
                ax[i].plot(DOAs, NMSE_DOA[3], marker=(5,2), linestyle="none", mfc="none", label="GRU Beamformer")
                ax[i].plot(DOAs, NMSE_DOA[4], marker="x", linestyle="none", mfc="none", label="CapsFormer")
                ax[i].plot(DOAs, ESE_DOA[1], marker="+", linestyle="none", mfc="none", label="Optimal")
                ax[i].set_title('NMSE DOAs')
                ax[i].set_ylabel('SE-NMSE')
                ax[i].set_xlabel('DOA')
                #ax[i].legend()
                ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)
            else:
                ax[i].plot(SNRs[:,i-1], NMSE[0][i-1], marker="D", linestyle="none", mfc="none", label="MVDR Beamformer")
                ax[i].plot(SNRs[:,i-1], NMSE[1][i-1], marker="o", linestyle="none", mfc="none", label="MMSE Beamformer")
                ax[i].plot(SNRs[:,i-1], NMSE[2][i-1], marker="s", linestyle="none", mfc="none", label="Deep INCM Reconst Net")
                ax[i].plot(SNRs[:,i-1], NMSE[3][i-1], marker=(5,2), linestyle="none", mfc="none", label="GRU Beamformer")
                ax[i].plot(SNRs[:,i-1], NMSE[4][i-1], marker="x", linestyle="none", mfc="none", label="CapsFormer")
                ax[i].plot(SNRs[:,i-1], ESE[1][i-1], marker="+", linestyle="none", mfc="none", label="Optimal")
                ax[i].set_title(f'NMSE DOA={DOAs[i-1]}')
                ax[i].set_ylabel('SE-NMSE')
                ax[i].set_xlabel('SNR')
                #ax[i].legend()
                ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)

        tikzplotlib.save(test_path + "/" + "DOA=" + str(DOAs) + "_DOA_SOI_perturb=" + str(DOAs_SOI_perturb) + "_test_NMSE_plot.tex")
        fig.savefig(test_path + "/" + "DOA=" + str(DOAs)+ "_DOA_SOI_perturb=" + str(DOAs_SOI_perturb) + "_test_NMSE_plot.pdf", bbox_inches="tight")
    
    def plot_BIAS(self, BIAS_DOA, BIAS, DOAs, DOAs_SOI_perturb, SNRs, test_path):
        """
        """
        fig, ax = plt.subplots(len(DOAs) + 1, 1, figsize=(6, (len(DOAs) + 1) * 6))

        for i in range(len(DOAs) + 1):
            if i == 0:
                ax[i].plot(DOAs, BIAS_DOA[0], marker="D", linestyle="none", mfc="none", label="MVDR Beamformer")
                ax[i].plot(DOAs, BIAS_DOA[1], marker="o", linestyle="none", mfc="none", label="MMSE Beamformer")
                ax[i].plot(DOAs, BIAS_DOA[2], marker="s", linestyle="none", mfc="none", label="Deep INCM Reconst Net")
                ax[i].plot(DOAs, BIAS_DOA[3], marker=(5,2), linestyle="none", mfc="none", label="GRU Beamformer")
                ax[i].plot(DOAs, BIAS_DOA[4], marker="x", linestyle="none", mfc="none", label="CapsFormer")
                ax[i].set_title('Bias DOAs')
                ax[i].set_ylabel('Bias')
                ax[i].set_xlabel('DOA')
                #ax[i].legend()
                ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)
            else:
                ax[i].plot(SNRs[:,i-1], BIAS[0][i-1], marker="D", linestyle="none", mfc="none", label="MVDR Beamformer")
                ax[i].plot(SNRs[:,i-1], BIAS[1][i-1], marker="o", linestyle="none", mfc="none", label="MMSE Beamformer")
                ax[i].plot(SNRs[:,i-1], BIAS[2][i-1], marker="s", linestyle="none", mfc="none", label="Deep INCM Reconst Net")
                ax[i].plot(SNRs[:,i-1], BIAS[3][i-1], marker=(5,2), linestyle="none", mfc="none", label="GRU Beamformer")
                ax[i].plot(SNRs[:,i-1], BIAS[4][i-1], marker="x", linestyle="none", mfc="none", label="CapsFormer")
                ax[i].set_title(f'Bias DOA={DOAs[i-1]}')
                ax[i].set_ylabel('Bias')
                ax[i].set_xlabel('SNR')
                #ax[i].legend()
                ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)

        tikzplotlib.save(test_path + "/" + "DOA=" + str(DOAs) + "_DOA_SOI_perturb=" + str(DOAs_SOI_perturb) + "_test_BIAS_plot.tex")
        fig.savefig(test_path + "/" + "DOA=" + str(DOAs) + "_DOA_SOI_perturb=" + str(DOAs_SOI_perturb) + "_test_BIAS_plot.pdf", bbox_inches="tight")

    def compare_beampattern(self, DOA, DOA_SOI_true, DOA_SOI_perturb, SNR, rng=None):
        """ 
        """
        arr = ULA(self.M)

        if rng is None:
            rng = np.random.default_rng(self.random_state)

        X, S = arr.array_response(self.T, DOA, len(DOA), SNR, rng=rng)
        if DOA_SOI_true is not None:
            #S = S[np.where(np.isin(DOA, DOA_SOI))[0]]
            SNR_i = SNR[np.where(np.isin(DOA, DOA_SOI_true))[0]]
            K = len(DOA_SOI_true)
            if DOA_SOI_perturb:
                DOA_SOI_est = np.clip(DOA_SOI_true + rng.uniform(-2.5, 2.5, K), -90, 90)
            else:
                DOA_SOI_est = DOA_SOI_true
        else:
            DOA_SOI_est = DOA_SOI_true
            SNR_i = SNR
            K = len(DOA)
        
        w_mvdr = arr.estimate_mvdr(X, K, DOA_SOI_est)
        w_mmse = arr.estimate_mmse(X, K, 10**(SNR_i/10), DOA_SOI_est)
        w_dirn = arr.estimate_dirn(X, K, self.model_dir, DOA_SOI_est)
        if DOA_SOI_true is not None:
            w_rnnbf = arr.estimate_rnnbf(X, K, self.model_dir, np.insert(np.delete(DOA, np.where(np.isin(DOA, DOA_SOI_true))[0]), 0, DOA_SOI_est))
        else:
            w_rnnbf = arr.estimate_rnnbf(X, K, self.model_dir, DOA_SOI_est)
        w_dbf, _, _ = arr.estimate_dbf(X, K, self.model_dir, DOA_SOI_est)
        w_opt = np.zeros((X.shape[0], K)).astype('complex64')

        qamma = 10**(SNR/10)
        A = arr.steering_vector(DOA)

        if DOA_SOI_true is not None:
            for i in range(len(DOA_SOI_true)):
                j = np.where(np.isin(DOA, DOA_SOI_true[i]))[0]
                a = arr.steering_vector(DOA_SOI_true[i])
                Q = np.delete(A, j, axis=1) @ np.diag(np.delete(qamma, j)) @ np.delete(A, j, axis=1).conj().T + np.eye(A.shape[0])
                iQ = np.linalg.solve(Q, np.eye(A.shape[0]))
                w_opt[:,i] = (iQ @ a / (a.conj().T @ iQ @ a)).reshape(-1)
        else:
            for i in range(len(DOA)):
                a = arr.steering_vector(DOA[i])
                Q = np.delete(A, i, axis=1) @ np.diag(np.delete(qamma, i)) @ np.delete(A, i, axis=1).conj().T + np.eye(A.shape[0])
                iQ = np.linalg.solve(Q, np.eye(A.shape[0]))
                w_opt[:,i] = (iQ @ a / (a.conj().T @ iQ @ a)).reshape(-1)

        Gr_size = 181
        dtheta= 180/(Gr_size-1)          
        angle_grid = np.arange(-90,90,dtheta)
        aa = arr.steering_vector(angle_grid)

        beampattern_mvdr = np.abs(w_mvdr.conj().T @ aa)**2
        beampattern_mmse = np.abs(w_mmse.conj().T @ aa)**2
        beampattern_dirn = np.abs(w_dirn.conj().T @ aa)**2
        beampattern_rnnbf = np.abs(w_rnnbf.conj().T @ aa)**2
        beampattern_dbf = np.abs(w_dbf.conj().T @ aa)**2
        beampattern_opt = np.abs(w_opt.conj().T @ aa)**2
        beampatterns = np.stack([beampattern_mvdr, beampattern_mmse, beampattern_dirn, beampattern_rnnbf, beampattern_dbf, beampattern_opt], axis=1)

        test_path = os.path.abspath(self.test_dir)

        if not os.path.exists(test_path):
            os.makedirs(test_path)
        
        if DOA_SOI_true is not None:
            for i in range(len(DOA_SOI_true)):
                self.plot_beampattern(angle_grid, beampatterns[i], DOA, DOA_SOI_true[i], DOA_SOI_est[i], DOA_SOI_perturb, SNR_i[i], test_path)
        else:
            for i in range(len(DOA)):
                self.plot_beampattern(angle_grid, beampatterns[i], DOA, DOA[i], None, DOA_SOI_perturb, SNR_i[i], test_path)
        
    def plot_beampattern(self, angle_grid, beampatterns, DOA, DOA_SOI_true, DOA_SOI_est, DOA_SOI_perturb, SNR, test_path):
        """
        """
        fig = plt.figure(figsize=(6,6))
        plt.title(f"DOA: {DOA}, DOA_SOI_true: {DOA_SOI_true}, DOA_SOI_est: {DOA_SOI_est}, SNR: {SNR}")
        plt.plot(angle_grid, 10*np.log10(np.abs(beampatterns[0])), label="MVDR")
        plt.plot(angle_grid, 10*np.log10(np.abs(beampatterns[1])), label="MMSE")
        plt.plot(angle_grid, 10*np.log10(np.abs(beampatterns[2])), label="Deep INCM Reconst Net")
        plt.plot(angle_grid, 10*np.log10(np.abs(beampatterns[3])), label="GRU Beamformer")
        plt.plot(angle_grid, 10*np.log10(np.abs(beampatterns[4])), label="CapsFormer")
        plt.plot(angle_grid, 10*np.log10(np.abs(beampatterns[5])), label="Optimal")
        plt.vlines(np.setdiff1d(DOA, DOA_SOI_true), 10*np.log10(np.abs(beampatterns)).min(), 10*np.log10(np.abs(beampatterns)).max(), colors="b", linestyles="--", label="Interferer DOA")
        plt.vlines(DOA_SOI_true, 10*np.log10(np.abs(beampatterns)).min(), 10*np.log10(np.abs(beampatterns)).max(), colors="r", linestyles=":", label="SOI DOA")
        #plt.legend()

        tikzplotlib.save(test_path + "/" + "DOA=" + str(DOA_SOI_true) + "_DOA_perturb=" + str(DOA_SOI_perturb) + "_test_beampattern_plot.tex")
        fig.savefig(test_path + "/" + "DOA=" + str(DOA_SOI_true) + "_DOA_perturb=" + str(DOA_SOI_perturb) + "_test_beampattern_plot.pdf", bbox_inches="tight")

