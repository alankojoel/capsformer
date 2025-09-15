import numpy as np
from tqdm import tqdm
import os
import torch
from src.models import ULA

class DataGenerator:
    """
    A class to generate and save training and validation data.

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
        self.random_state = kwargs["data_random_state"]
        self.n_samples = kwargs["n_samples"]
        self.data_dir = kwargs["data_dir"]

    def generate_data(self, n_samples=None):
        """
        """
        if n_samples is None:
            n_samples = self.n_samples
        
        arr = ULA(self.M)

        SNRs = np.arange(self.SNR_start, self.SNR_end+1, self.SNR_spacing)
        rng = np.random.default_rng(self.random_state)

        SOI_train = []
        DOA_train = []
        X_train = []
        SCM_train = []

        pbar_train = tqdm(total=len(SNRs) * n_samples, leave=True)

        for j in range(len(SNRs)):

            #SNR_train = SNRs[j]

            for i in range(n_samples):

                SNR_train = rng.choice(SNRs, self.K)

                while True:
                    DOA_SOI_train = rng.uniform(self.DOA_SOI_start, self.DOA_SOI_end, size=self.K)
                    if self.K == 1:
                        break
                    DOA_SOI_train = np.sort(DOA_SOI_train)
                    doa_diffs = np.diff(DOA_SOI_train)
                    if np.all(doa_diffs >= 10):
                        break

                X, S = arr.array_response(self.T, DOA_SOI_train, self.K, SNR_train, rng)
                #X = X / np.linalg.norm(X, axis=1, keepdims=True)
                SCM = arr.compute_SCM(X)
                SCM = self.M * SCM / np.trace(SCM)

                S = np.concatenate([S.real, S.imag], axis=1)
                X = np.stack([X.real, X.imag], axis=0)
                SCM = np.stack([SCM.real, SCM.imag, np.angle(SCM)], axis=0) #np.stack([(SCM.real - SCM.real.mean())/SCM.real.std(), (SCM.imag - SCM.imag.mean())/SCM.imag.std(), np.angle(SCM)], axis=0) # np.angle(SCM)

                SOI_train.append(S)
                DOA_train.append(DOA_SOI_train)
                X_train.append(X)
                SCM_train.append(SCM)

                desc = f"Generating training data: DOA: {DOA_SOI_train}, SNR: {SNR_train}"
                pbar_train.set_description(desc)
                pbar_train.update()
        
        SOI_train = torch.tensor(np.array(SOI_train), dtype=torch.float32)
        DOA_train = torch.tensor(np.array(DOA_train), dtype=torch.float32)
        X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
        SCM_train = torch.tensor(np.array(SCM_train), dtype=torch.float32)

        self.save_data(X_train=X_train, SOI_train=SOI_train, DOA_train=DOA_train, SCM_train=SCM_train)    
        
        del X_train, SOI_train, DOA_train, SCM_train

        SOI_val = []
        DOA_val = []
        X_val = []
        SCM_val = []
        
        pbar_val = tqdm(total=n_samples, leave=True)

        for i in range(n_samples):

            SNR_val = rng.choice(SNRs, self.K)

            while True:
                DOA_SOI_val = rng.uniform(self.DOA_SOI_start, self.DOA_SOI_end, size=self.K)
                if self.K == 1:
                    break
                DOA_SOI_val = np.sort(DOA_SOI_val)
                doa_diffs = np.diff(DOA_SOI_val)
                if np.all(doa_diffs >= 10):
                    break

            X, S = arr.array_response(self.T, DOA_SOI_val, self.K, SNR_val, rng)
            #X = X / np.linalg.norm(X, axis=1, keepdims=True)
            SCM = arr.compute_SCM(X)
            SCM = self.M * SCM / np.trace(SCM)

            S = np.concatenate([S.real, S.imag], axis=1)
            X = np.stack([X.real, X.imag], axis=0)
            SCM = np.stack([SCM.real, SCM.imag, np.angle(SCM)], axis=0) #np.stack([(SCM.real - SCM.real.mean())/SCM.real.std(), (SCM.imag - SCM.imag.mean())/SCM.imag.std(), np.angle(SCM)], axis=0) # np.angle(SCM)

            SOI_val.append(S)
            DOA_val.append(DOA_SOI_val)
            X_val.append(X)
            SCM_val.append(SCM)

            desc = f"Generating validation data: DOA: {DOA_SOI_val}, SNR: {SNR_val}"
            pbar_val.set_description(desc)
            pbar_val.update()

        SOI_val = torch.tensor(np.array(SOI_val), dtype=torch.float32)
        DOA_val = torch.tensor(np.array(DOA_val), dtype=torch.float32)
        X_val = torch.tensor(np.array(X_val), dtype=torch.float32)
        SCM_val = torch.tensor(np.array(SCM_val), dtype=torch.float32)

        self.save_data(X_val=X_val, SOI_val=SOI_val, DOA_val=DOA_val, SCM_val=SCM_val)

        del X_val, SOI_val, DOA_val, SCM_val
    
    def data_id(self):
        """
        """
        return "_M=" + str(self.M) #"_".join([str(self.M), str(self.K), str(self.T)])
    
    def save_data(self, **kwargs):
        """ 
        """
        id = self.data_id()
        data_path = os.path.abspath(self.data_dir)

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        for fname, data in kwargs.items():
           torch.save(data, data_path + "/" + fname + id + ".pt") 



