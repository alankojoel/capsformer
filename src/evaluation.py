
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
    A class for evaluating the performance of the CapsFormer

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
        SOI_est = np.empty((3, len(SNR)), dtype=object)
        SOI_true = np.empty(len(SNR), dtype=object)

        for i, SNR_i in enumerate(SNR):
            X, S = arr.array_response(self.T, DOA, len(DOA), SNR_i, rng=rng)
            if DOA_SOI_true is not None:
                SOI_true[i] = S[np.where(np.isin(DOA, DOA_SOI_true))[0]]
                SNR_i = SNR_i[np.where(np.isin(DOA, DOA_SOI_true))[0]]
            else:
                SOI_true[i] = S
            SOI_est[0][i] = arr.estimate_capon(X, K, DOA_SOI_est)
            SOI_est[1][i] = arr.estimate_mmse(X, K, 10**(SNR_i/10), DOA_SOI_est)
            SOI_est[2][i], _, _ = arr.estimate_dbf(X, K, self.model_dir, DOA_SOI_est)

        return SOI_est, SOI_true
    
    def estimate_single_sample(self, arr, DOAs, DOAs_SOI, DOAs_SOI_perturb, SNRs, rng=None):
        """
        """
        if rng is None:
            rng = np.random.default_rng(self.random_state)

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
            SOI_i_est, SOI_i_true = self.estimate_SOI(arr, DOA_i, K, DOA_SOI_i_true, DOA_SOI_i_est, SNRs, rng=rng)
            SOIs_est.append(SOI_i_est)
            SOIs_true.append(SOI_i_true)

        SOIs_est = np.array(SOIs_est)
        SOIs_true = np.array(SOIs_true)

        NMSE = self.compute_NMSE(SNRs, SOIs_true, SOIs_est)
        BIAS = self.compute_BIAS(SNRs, SOIs_true, SOIs_est)

        return NMSE, BIAS

    
    def compute_NMSE(self, SNRs, SOIs_true, SOIs_est):
        """
        """
        NMSE =  [[] for _ in range(3)]
        
        num_DOA = SOIs_true.shape[0] if SOIs_true[0][0].shape[0] == 1 else SOIs_true[0][0].shape[0]
        
        if SOIs_true.shape[0] > 1:
            assert SOIs_true[0][0].shape[0] == 1
            
        for i in range(3): 
            NMSE[i]= np.zeros((num_DOA, len(SNRs)))

        for i in range(SOIs_true.shape[0]):
            for j in range(len(SNRs)):
                SOI_true = SOIs_true[i][j]
                if SOIs_true[0][0].shape[0] == 1:
                    NMSE[0][i][j] = (np.mean(np.abs(SOIs_est[i][0][j] - SOI_true)**2, axis=1) / np.mean(np.abs(SOI_true)**2, axis=1)).item()
                    NMSE[1][i][j] = (np.mean(np.abs(SOIs_est[i][1][j] - SOI_true)**2, axis=1) / np.mean(np.abs(SOI_true)**2, axis=1)).item()
                    NMSE[2][i][j] = (np.mean(np.abs(SOIs_est[i][2][j] - SOI_true)**2, axis=1) / np.mean(np.abs(SOI_true)**2, axis=1)).item()
                else:
                    NMSE[0][:,j] = np.mean(np.abs(SOIs_est[i][0][j] - SOI_true)**2, axis=1) / np.mean(np.abs(SOI_true)**2, axis=1)
                    NMSE[1][:,j] = np.mean(np.abs(SOIs_est[i][1][j] - SOI_true)**2, axis=1) / np.mean(np.abs(SOI_true)**2, axis=1)
                    NMSE[2][:,j] = np.mean(np.abs(SOIs_est[i][2][j] - SOI_true)**2, axis=1) / np.mean(np.abs(SOI_true)**2, axis=1)

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
                    ESE[0][i][j] = arr.capon_ESE(DOA_i, qamma_j) / qamma_j
                    ESE[1][i][j] = arr.mmse_ESE(DOA_i, qamma_j) / qamma_j
                else:
                    ESE[0][:,j] = arr.capon_ESE(DOA_i, qamma_j) / qamma_j
                    ESE[1][:,j] = arr.mmse_ESE(DOA_i, qamma_j) / qamma_j

        ESE = np.array(ESE)

        if DOAs_SOI is not None:
            ESE = ESE[:, np.where(np.isin(DOAs.flatten(), DOAs_SOI.flatten()))[0],:]
                    
        return ESE
        

    def compute_BIAS(self, SNRs, SOIs_true, SOIs_est):
        """
        """
        BIAS =  [[] for _ in range(3)]

        num_DOA = SOIs_true.shape[0] if  SOIs_true[0][0].shape[0] == 1 else SOIs_true[0][0].shape[0]
        if SOIs_true.shape[0] > 1:
            assert SOIs_true[0][0].shape[0] == 1
            
        for i in range(3): 
            BIAS[i]= np.zeros((num_DOA, len(SNRs)))

        for i in range(SOIs_true.shape[0]):
            for j in range(len(SNRs)):
                SOI_true = SOIs_true[i][j]
                if SOIs_true[0][0].shape[0] == 1:
                    BIAS[0][i][j] = ((np.mean(np.abs(SOIs_est[i][0][j])**2, axis=1) - np.mean(np.abs(SOI_true)**2, axis=1)) / np.mean(np.abs(SOI_true)**2, axis=1)).item()
                    BIAS[1][i][j] = ((np.mean(np.abs(SOIs_est[i][1][j])**2, axis=1) - np.mean(np.abs(SOI_true)**2, axis=1)) / np.mean(np.abs(SOI_true)**2, axis=1)).item()
                    BIAS[2][i][j] = ((np.mean(np.abs(SOIs_est[i][2][j])**2, axis=1) - np.mean(np.abs(SOI_true)**2, axis=1)) / np.mean(np.abs(SOI_true)**2, axis=1)).item()
                else:
                    BIAS[0][:,j] = (np.mean(np.abs(SOIs_est[i][0][j])**2, axis=1) - np.mean(np.abs(SOI_true)**2, axis=1)) / np.mean(np.abs(SOI_true)**2, axis=1)
                    BIAS[1][:,j] = (np.mean(np.abs(SOIs_est[i][1][j])**2, axis=1) - np.mean(np.abs(SOI_true)**2, axis=1)) / np.mean(np.abs(SOI_true)**2, axis=1)
                    BIAS[2][:,j] = (np.mean(np.abs(SOIs_est[i][2][j])**2, axis=1) - np.mean(np.abs(SOI_true)**2, axis=1)) / np.mean(np.abs(SOI_true)**2, axis=1)

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
            NMSE = np.zeros((3, DOAs.shape[0], len(SNRs)))
            BIAS = np.zeros((3, DOAs.shape[0], len(SNRs)))
        elif mode == "m":
            if DOAs_SOI is not None:
                assert np.all(np.isin(DOAs_SOI, DOAs))
                K = len(DOAs_SOI)
                DOAs_SOI = DOAs_SOI.reshape(1,-1)
            else:
                K = len(DOAs)
            DOAs = DOAs.reshape(1,-1)        
            NMSE = np.zeros((3, K, len(SNRs)))
            BIAS = np.zeros((3, K, len(SNRs)))
        else:
            raise Exception("Invalid mode.")
            
        pbar = tqdm(total=MC_trials, position=0, leave=True)

        for i in range(MC_trials):
            NMSE_i, BIAS_i = self.estimate_single_sample(arr, DOAs, DOAs_SOI, DOAs_SOI_perturb, SNRs, rng=rng)
            NMSE += NMSE_i
            BIAS += BIAS_i
            curr_NMSE_mean = np.mean(NMSE / (i + 1), axis=(1,2))
            curr_BIAS_mean = np.mean(BIAS / (i + 1), axis=(1,2))
            DOA_s = ",".join(f"{DOAs[i]}" for i in range(DOAs.shape[0]))
            DOA_SOI_s = ",".join(f"{DOAs_SOI[i]}" for i in range(DOAs_SOI.shape[0])) if DOAs_SOI is not None else DOA_s
            SNR_s = ",".join(f"{SNRs[i]}" for i in range(SNRs.shape[0]))
            desc = f'DOA: {DOA_s}, DOA_SOI: {DOA_SOI_s}, Apply perturbations: {DOAs_SOI_perturb}, MC trials: {i + 1}, Average NMSE: Capon Beamformer: {curr_NMSE_mean[0]:.4f}, MMSE Beamformer: {curr_NMSE_mean[1]:.4f}, Deep Beamformer: {curr_NMSE_mean[2]:.4f}, Average BIAS: Capon Beamformer: {curr_BIAS_mean[0]:.4f}, MMSE Beamformer: {curr_BIAS_mean[1]:.4f} Deep Beamformer: {curr_BIAS_mean[2]:.4f}' #SNR: {SNR_string}, 
            pbar.set_description(desc)  
            pbar.update()

        pbar.close()

        ESE = self.compute_ESE(arr, DOAs, DOAs_SOI, SNRs)

        return NMSE / MC_trials, np.mean(NMSE / MC_trials, axis=2), ESE, np.mean(ESE, axis=2), BIAS / MC_trials, np.mean(BIAS / MC_trials, axis=2)
    
    def compare_metrics(self, mode, DOAs, DOAs_SOI, DOAs_SOI_perturb, SNRs, MC_trials=None):
        """ 
        """
        if MC_trials is None:
            MC_trials = self.MC_trials
        
        NMSE, NMSE_DOA, ESE, ESE_DOA, BIAS, BIAS_DOA = self.estimate_MC_trials(mode, DOAs, DOAs_SOI, DOAs_SOI_perturb, SNRs, MC_trials=MC_trials)
        
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
        
        log_stats(log_path, f"(NMSE DOAs) Capon Beamformer: {', '.join(f'{x:.4f}' for x in NMSE_DOA[0])}")
        log_stats(log_path, f"(NMSE DOAs) MMSE Beamformer: {', '.join(f'{x:.4f}' for x in NMSE_DOA[1])}")
        log_stats(log_path, f"(NMSE DOAs) Deep Beamformer: {', '.join(f'{x:.4f}' for x in NMSE_DOA[2])}")
        log_stats(log_path, f"(NMSE DOAs) Optimal: {', '.join(f'{x:.4f}' for x in ESE_DOA[1])}")

        for i in range(DOAs_act.shape[0]):
            log_stats(log_path, f"(NMSE SNRs DOA={DOAs_act[i]}) Capon Beamformer: {', '.join(f'{x:.4f}' for x in NMSE[0][i])}")
            log_stats(log_path, f"(NMSE SNRs DOA={DOAs_act[i]}) MMSE Beamformer: {', '.join(f'{x:.4f}' for x in NMSE[1][i])}")
            log_stats(log_path, f"(NMSE SNRs DOA={DOAs_act[i]}) Deep Beamformer: {', '.join(f'{x:.4f}' for x in NMSE[2][i])}")
            log_stats(log_path, f"(NMSE SNRs DOA={DOAs_act[i]}) Optimal: {', '.join(f'{x:.4f}' for x in ESE[1][i])}")

        log_stats(log_path, f"(BIAS DOAs) Capon Beamformer: {', '.join(f'{x:.4f}' for x in BIAS_DOA[0])}")
        log_stats(log_path, f"(BIAS DOAs) MMSE Beamformer: {', '.join(f'{x:.4f}' for x in BIAS_DOA[1])}")
        log_stats(log_path, f"(BIAS DOAs) Deep Beamformer: {', '.join(f'{x:.4f}' for x in BIAS_DOA[2])}")

        for i in range(DOAs_act.shape[0]):
            log_stats(log_path, f"(BIAS SNRs DOA={DOAs_act[i]}) Capon Beamformer: {', '.join(f'{x:.4f}' for x in BIAS[0][i])}")
            log_stats(log_path, f"(BIAS SNRs DOA={DOAs_act[i]}) MMSE Beamformer: {', '.join(f'{x:.4f}' for x in BIAS[1][i])}")
            log_stats(log_path, f"(BIAS SNRs DOA={DOAs_act[i]}) Deep Beamformer: {', '.join(f'{x:.4f}' for x in BIAS[2][i])}")

        self.plot_NMSE(NMSE_DOA, NMSE, ESE, ESE_DOA, DOAs_act, DOAs_SOI_perturb, SNRs_act, test_path)

        self.plot_BIAS(BIAS_DOA, BIAS, DOAs_act, DOAs_SOI_perturb, SNRs_act, test_path)
    
    def plot_NMSE(self, NMSE_DOA, NMSE, ESE, ESE_DOA, DOAs, DOAs_SOI_perturb, SNRs, test_path):
        """
        """
        fig, ax = plt.subplots(len(DOAs) + 1, 1, figsize=(6, (len(DOAs) + 1) * 6))

        for i in range(len(DOAs) + 1):
            if i == 0:
                ax[i].plot(DOAs, NMSE_DOA[0], marker="D", linestyle="none", label="Capon Beamformer")
                ax[i].plot(DOAs, NMSE_DOA[1], marker="o", linestyle="none", label="MMSE Beamformer")
                ax[i].plot(DOAs, NMSE_DOA[2], marker="s", linestyle="none", label="Deep Beamformer")
                ax[i].plot(DOAs, ESE_DOA[1], marker="+", linestyle="none", label="Optimal")
                ax[i].set_title('NMSE DOAs')
                ax[i].set_ylabel('SE-NMSE')
                ax[i].set_xlabel('DOA')
                #ax[i].legend()
                ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)
            else:
                ax[i].plot(SNRs[:,i-1], NMSE[0][i-1], marker="D", linestyle="none", label="Capon Beamformer")
                ax[i].plot(SNRs[:,i-1], NMSE[1][i-1], marker="o", linestyle="none", label="MMSE Beamformer")
                ax[i].plot(SNRs[:,i-1], NMSE[2][i-1], marker="s", linestyle="none", label="Deep Beamformer")
                ax[i].plot(SNRs[:,i-1], ESE[1][i-1], marker="+", linestyle="none", label="Optimal")
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
                ax[i].plot(DOAs, BIAS_DOA[0], marker="D", linestyle="none", label="Capon Beamformer")
                ax[i].plot(DOAs, BIAS_DOA[1], marker="o", linestyle="none", label="MMSE Beamformer")
                ax[i].plot(DOAs, BIAS_DOA[2], marker="s", linestyle="none", label="Deep Beamformer")
                ax[i].set_title('Bias DOAs')
                ax[i].set_ylabel('Bias')
                ax[i].set_xlabel('DOA')
                #ax[i].legend()
                ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)
            else:
                ax[i].plot(SNRs[:,i-1], BIAS[0][i-1], marker="D", linestyle="none", label="Capon Beamformer")
                ax[i].plot(SNRs[:,i-1], BIAS[1][i-1], marker="o", linestyle="none", label="MMSE Beamformer")
                ax[i].plot(SNRs[:,i-1], BIAS[2][i-1], marker="s", linestyle="none", label="Deep Beamformer")
                ax[i].set_title(f'Bias DOA={DOAs[i-1]}')
                ax[i].set_ylabel('Bias')
                ax[i].set_xlabel('SNR')
                #ax[i].legend()
                ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)

        tikzplotlib.save(test_path + "/" + "DOA=" + str(DOAs) + "_DOA_SOI_perturb=" + str(DOAs_SOI_perturb) + "_test_BIAS_plot.tex")
        fig.savefig(test_path + "/" + "DOA=" + str(DOAs) + "_DOA_SOI_perturb=" + str(DOAs_SOI_perturb) + "_test_BIAS_plot.pdf", bbox_inches="tight")

    
    def compare_SOI(self, DOA, SNR, rng=None):
        """ 
        """
        arr = ULA(self.M)

        if rng is None:
            rng = np.random.default_rng(self.random_state)

        X, S = arr.array_response(self.T, DOA, len(DOA), SNR, rng=rng)
        S_capon = arr.estimate_capon(X, len(DOA), np.array(DOA))
        S_dbf, DOA_est, DOA_dist = arr.estimate_dbf(X, len(DOA), self.model_dir)

        test_path = os.path.abspath(self.test_dir)

        if not os.path.exists(test_path):
            os.makedirs(test_path)
        
        self.plot_DOA(DOA_dist, DOA_est, DOA, SNR, test_path)

        for i in range(len(DOA)):
            self.plot_SOI(S[i], S_capon[i], S_dbf[i], DOA[i], SNR[i], test_path)
    
    def plot_DOA(self, DOA_dist, DOA_est, DOA_true, SNR, test_path):
        """
        """
        fig = plt.figure(figsize=(6,6))
        plt.plot(np.arange(-90, 90+1), DOA_dist)
        plt.vlines(DOA_est, 0, DOA_dist[DOA_est + 90], colors="b", linestyles="--", label="Predicted DOA")
        plt.vlines(np.round(DOA_true).astype('int'), 0, DOA_dist[np.round(DOA_true).astype('int') + 90], colors="r", linestyles=":", label="True DOA")
        #plt.legend()
        plt.title(f"Predicted DOA: {DOA_est}, True DOA: {DOA_true}, SNR: {SNR}")
        tikzplotlib.save(test_path + "/" + "DOA=" + str(DOA_true) + "_test_DOA_plot.tex")
        fig.savefig(test_path + "/" + "DOA=" + str(DOA_est) + "_test_DOA_plot.pdf", bbox_inches="tight")


    def plot_SOI(self, S_true, S_capon, S_dbf, DOA, SNR, test_path):
        """
        """
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f"True and predicted SOI with DOA={DOA} and SNR={SNR}")
    
        ax[0,0].plot(S_true.real.T, label="True")
        ax[0,1].plot(S_true.imag.T, label="True")

        ax[0,0].plot(S_capon.real.T, label="Capon estimate")
        ax[0,1].plot(S_capon.imag.T, label="Capon estimate")

        ax[1,0].plot(S_true.real.T, label="True")
        ax[1,1].plot(S_true.imag.T, label="True")

        ax[1,0].plot(S_dbf.real.T, label="DBF estimate")
        ax[1,1].plot(S_dbf.imag.T, label="DBF estimate")
        
        for i in range(2):
            for j in range(2):
                if j == 0:
                    ax[i,j].set_title('Re Part')
                else:
                    ax[i,j].set_title('Im Part')
                ax[i,j].set_ylabel('Value')
                ax[i,j].set_xlabel('Sample')
                #ax[i,j].legend()
                ax[i,j].grid(True, which='both', linestyle='--', linewidth=0.5)
        
        tikzplotlib.save(test_path + "/" + "DOA=" + str(DOA) + "_test_SOI_plot.tex")
        fig.savefig(test_path + "/" + "DOA=" + str(DOA) + "_test_SOI_plot.pdf", bbox_inches="tight")

