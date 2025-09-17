import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Laplace
from torch.autograd import Variable
import gc
from src.utils import log_stats
from src.models import DeepBeamformer

class ArrayDataset(Dataset):
    """
    Custom dataset class for handling training and validation data.

    """
    def __init__(self, Xs, SCMs, SOIs, DOAs):
        self.Xs = Xs
        self.SCMs = SCMs
        self.SOIs = SOIs
        self.DOAs = DOAs

    def __len__(self):
        """
        """
        return len(self.Xs)

    def __getitem__(self, idx):
        """
        """
        X_i = self.Xs[idx]
        SCM_i = self.SCMs[idx]
        SOI_i = self.SOIs[idx]
        DOA_i = self.DOAs[idx]
        DOA_i_r = torch.round(DOA_i).to(torch.int64) + 90
        DOA_i_k = torch.zeros((DOA_i_r.shape[0], 181)).scatter_(1, DOA_i_r.view(-1, 1), 1.).sum(dim=0) #torch.nn.functional.one_hot(DOA_i, num_classes=181).to(torch.float32)
        
        return X_i, SCM_i, SOI_i, DOA_i, DOA_i_k
        

class NMSELoss(nn.Module):
    """
    Custom loss function for computing the normalized mean square signal reconstruction error.

    """
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, input, target):
        power = torch.sum(target ** 2, axis=-1, keepdims=True)
        loss = torch.sum((input - target) ** 2, axis=-1, keepdims=True)
        loss = loss / (power + 1e-8)
        loss = torch.mean(loss)
        return loss

class NMSEPLoss(nn.Module):
    """
    Custom loss function for computing the normalized mean square signal power error.
    
    """
    def __init__(self):
        super(NMSEPLoss, self).__init__()

    def forward(self, input, target):
        power_input = torch.sum(input ** 2, axis=-1, keepdims=True)
        power_target = torch.sum(target ** 2, axis=-1, keepdims=True)
        loss = (power_input - power_target) ** 2
        loss = loss / (power_target + 1e-8)
        loss = torch.mean(loss)
        return loss

class Trainer:
    """
    A class to manage the training process for CapsFormer.

    """
    def __init__(self, device, **kwargs):
        self.device = device
        self.kwargs = kwargs
        self.M = kwargs["M"]
        self.K = kwargs["K"]
        self.T = kwargs["T"]
        self.random_state = kwargs["train_random_state"]
        self.num_epochs = kwargs["num_epochs"]
        self.batch_size = kwargs["batch_size"]
        self.scale = kwargs["scale"]
        self.wd = kwargs["wd"]
        self.lr = kwargs["lr"]
        self.lr_gamma = kwargs["lr_gamma"]
        self.lr_milestones = kwargs["lr_milestones"]
        self.data_dir = kwargs["data_dir"]
        self.model_dir = kwargs["model_dir"]
        self.train_dir = kwargs["train_dir"]

        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.optimizer_lr_sched = None
        self.model = None

        self.model_path = None
        self.train_path = None
        self.log_path = None
        self.best_snapshot_fname = None
        self.best_metric = None

    def init_model(self):
        """
        """
        if hasattr(self, "model"):
            del self.model
            gc.collect()

        self.model = DeepBeamformer(self.M).to(self.device)

        """
        for layer in self.model.modules():
            if isinstance(layer, (nn.Linear)):
                nn.init.normal_(layer.weight, mean=0, std=0.25)
        """

    def data_id(self):
        """
        """
        return "_M=" + str(self.M) #"_".join([str(self.model_cfg.K), str(self.model_cfg.M), str(self.model_cfg.T)])

    def load_data(self):
        """
        """
        id = self.data_id()
        data_path = os.path.abspath(self.data_dir)
        
        try:
            X_train = torch.load(data_path + "/X_train" + id + ".pt")
        except FileNotFoundError:
            raise Exception("Load data: training feature set doesn't exist")
        try:
            SCM_train = torch.load(data_path + "/SCM_train" + id + ".pt")
        except FileNotFoundError:
            raise Exception("Load data: training feature set doesn't exist")
        try:
            SOI_train = torch.load(data_path + "/SOI_train" + id + ".pt")
        except FileNotFoundError:
            raise Exception("Load data: training label set doesn't exist")
        try:
            DOA_train = torch.load(data_path + "/DOA_train" + id + ".pt")
        except FileNotFoundError:
            raise Exception("Load data: training label set doesn't exist")
        
        try:
            X_val = torch.load(data_path + "/X_val" + id + ".pt")
        except FileNotFoundError:
            raise Exception("Load data: validation feature set doesn't exist")
        try:
            SCM_val = torch.load(data_path + "/SCM_val" + id + ".pt")
        except FileNotFoundError:
            raise Exception("Load data: validation feature set doesn't exist")
        try:
            SOI_val = torch.load(data_path + "/SOI_val" + id + ".pt")
        except FileNotFoundError:
            raise Exception("Load data: validation label set doesn't exist")
        try:
            DOA_val = torch.load(data_path + "/DOA_val" + id + ".pt")
        except FileNotFoundError:
            raise Exception("Load data: validation label set doesn't exist")

        train_set = ArrayDataset(X_train, SCM_train, SOI_train, DOA_train)
        val_set = ArrayDataset(X_val, SCM_val, SOI_val, DOA_val)


        return train_set, val_set

    def init_paths(self):
        """
        """
        del self.model_path
        del self.train_path
        del self.log_path

        self.model_path = os.path.abspath(self.model_dir)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        self.train_path = os.path.abspath(self.train_dir)
        if not os.path.exists(self.train_path):
            os.makedirs(self.train_path)

        self.log_path = os.path.join(self.train_path, "train.log")
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        
        with open(self.log_path, 'w') as fp:
            pass

    def init_run(self):
        """ 
        """
        del self.train_loader
        del self.val_loader
        del self.optimizer
        del self.optimizer_lr_sched

        gc.collect()

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.init_paths()
        self.init_model()
        train_ds, val_ds = self.load_data()

        self.train_loader = DataLoader(train_ds,
                                       batch_size=self.batch_size,
                                       shuffle=True)
        
        self.val_loader = DataLoader(val_ds,
                                     batch_size=self.batch_size,
                                     shuffle=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.wd)
        
        self.optimizer_lr_sched = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                       milestones=self.lr_milestones,
                                                                       gamma=self.lr_gamma)       

    def run(self, n_epochs=None):
        """ 
        """
        self.init_run()

        if n_epochs is None:
            n_epochs = self.num_epochs
        
        for self.epoch in range(n_epochs):

            self.model.train()
            train_loss, train_m_loss, train_r_loss, train_p_loss, train_acc, train_rmse = self.train_epoch()
            log_stats(self.log_path, f"[{self.epoch+1}] Train - Regression loss: {train_m_loss}, Reconstruction loss: {train_r_loss}, Power loss: {train_p_loss}, DOA accuracy: {train_acc}, DOA RMSE: {train_rmse}")

            self.model.eval()
            val_loss, val_m_loss, val_r_loss, val_p_loss, val_acc, val_rmse = self.val_epoch()
            log_stats(self.log_path, f"[{self.epoch+1}] Validation - Regression loss: {val_m_loss}, Reconstruction loss: {val_r_loss}, Power loss: {val_p_loss}, DOA accuracy: {val_acc}, DOA RMSE: {val_rmse}")

            if self.epoch + 1 <= 26: # 22 40
                self.scale *= 0.95

        id = self.data_id()
        torch.save(self.model.state_dict(), self.model_path + "/deepbeamformer" + id + ".pt")
    
    def loss_beamformer(self, X, SCM, SOI, DOA, DOA_k_hot, mode="train"):
        """
        """
        if mode == "train":
            DOA_pert = self.DOA_perturb(DOA)
            W_est, DOA_est = self.model(SCM, self.K, DOA_pert)
        elif mode == "val":
            W_est, DOA_est = self.model(SCM, self.K, DOA)
        else:
            raise Exception("Invalid loss mode.")

        W_est = W_est[:,:self.M,:] + 1j * W_est[:,self.M:,:]
        X = X[:,0,:,:] + 1j * X[:,1,:,:]
        SOI_est = torch.matmul(W_est.transpose(1,2).conj(), X)
        SOI_est = torch.cat((SOI_est.real, SOI_est.imag), dim=2)

        DOA_smooth = self.smooth_k_hot(DOA_k_hot, self.scale).to(DOA_k_hot.device)
            
        if (self.epoch == 0 or (self.epoch + 1) % 5 == 0) and self.i == 0 and mode == "train":
            self.plot_DOA(DOA_smooth[:4].cpu().detach().numpy(), DOA_est[:4].cpu().detach().numpy())
            
        loss_m = self.loss_regres(DOA_est, DOA_smooth)
        loss_r = self.loss_reconst(SOI_est, SOI)
        loss_p = self.loss_power(SOI_est, SOI)
        clf_acc = self.classification_accuracy(DOA_est, DOA_k_hot)
        clf_rmse = self.classification_RMSE(DOA_est, DOA)

        loss = loss_m + 1 * loss_r + 0.001 * loss_p

        return loss, loss_m, loss_r, loss_p, clf_acc, clf_rmse

    def DOA_perturb(self, DOA):
        """
        """
        dist = Laplace(0, self.scale)
        pert = dist.sample(DOA.shape).to(DOA.device)
        DOA_pert = torch.clamp(DOA + pert, -90, 90)

        return DOA_pert

    def loss_regres(self, DOA_est, DOA):
        """
        """
        return nn.KLDivLoss(reduction='batchmean')(DOA_est.log(), DOA)
        
        
    def loss_reconst(self, SOI_est, SOI):
        """
        """
        return NMSELoss()(SOI_est, SOI) 

    def loss_power(self, SOI_est, SOI):
        """
        """
        return NMSEPLoss()(SOI_est, SOI)

    def classification_accuracy(self, DOA_est, DOA):
        """
        """
        b = DOA.shape[0]
        DOA_est = self.model.find_top_k_peaks(DOA_est, self.K)
        DOA_est = torch.zeros((DOA_est.shape[0], 181), device=DOA_est.device).scatter_(1, DOA_est, 1.).to(torch.int8)
        DOA_true = DOA.to(torch.int8) 
        correct = (DOA_est & DOA_true).sum()

        return correct / (self.K * b)

    def classification_RMSE(self, DOA_est, DOA):
        """
        """
        DOA_est = self.model.find_top_k_peaks(DOA_est, self.K).float() - 90

        return torch.sqrt(nn.MSELoss()(DOA_est, DOA))

    def smooth_k_hot(self, k_hot, scale):
        """
        """
        B, C = k_hot.shape
        idxs = torch.arange(C).unsqueeze(0).repeat(C, 1) 
        centers = torch.arange(C).unsqueeze(1)            
        laplace_kernel = torch.exp(-torch.abs(idxs - centers) / scale)
        laplace_kernel = laplace_kernel / laplace_kernel.sum(dim=1, keepdim=True)
    
        smoothed = torch.matmul(k_hot, laplace_kernel.to(k_hot.device)) 
                
        return smoothed / smoothed.sum(dim=1, keepdim=True)
        
    def plot_DOA(self, DOA, DOA_est):
        """
        """
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f"True and predicted DOA distribution at epoch {self.epoch}")

        ax[0,0].plot(DOA[0], label="True DOA")
        ax[0,0].plot(DOA_est[0], label="Predicted DOA")

        ax[0,1].plot(DOA[1], label="True DOA")
        ax[0,1].plot(DOA_est[1], label="Predicted DOA")
        
        ax[1,0].plot(DOA[2], label="True DOA")
        ax[1,0].plot(DOA_est[2], label="Predicted DOA")
        
        ax[1,1].plot(DOA[3], label="True DOA")
        ax[1,1].plot(DOA_est[3], label="Predicted DOA")

        #tikzplotlib.save(self.train_path + "/" + "DOA_distributions_epoch_" + str(self.epoch) + ".tex")
        fig.savefig(self.train_path + "/" + "DOA_distributions_epoch_" + str(self.epoch) + ".pdf", bbox_inches="tight")

    def train_epoch(self):
        """
        """
        pbar = tqdm(total=len(self.train_loader), position=0, leave=True)

        running_loss = torch.tensor(0., requires_grad=False).to(self.device)
        running_m_loss = torch.tensor(0., requires_grad=False).to(self.device)
        running_r_loss = torch.tensor(0., requires_grad=False).to(self.device)
        running_p_loss = torch.tensor(0., requires_grad=False).to(self.device)
        running_acc = torch.tensor(0., requires_grad=False).to(self.device)
        running_rmse = torch.tensor(0., requires_grad=False).to(self.device)

        for self.i, (batch) in enumerate(self.train_loader):
            
            X, SCM, SOI, DOA, DOA_k_hot = batch
            X, SCM, SOI, DOA, DOA_k_hot = X.to(self.device), SCM.to(self.device), SOI.to(self.device), DOA.to(self.device), DOA_k_hot.to(self.device)
            
            loss, loss_m, loss_r, loss_p, acc, rmse = self.loss_beamformer(X, SCM, SOI, DOA, DOA_k_hot)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            cur_loss = running_loss.item() / (self.i + 1)

            running_m_loss += loss_m.item()
            cur_m_loss = running_m_loss.item() / (self.i + 1)

            running_r_loss += loss_r.item()
            cur_r_loss = running_r_loss.item() / (self.i + 1)

            running_p_loss += loss_p.item()
            cur_p_loss = running_p_loss.item() / (self.i + 1)
            
            running_acc += acc.item()
            cur_acc = running_acc.item() / (self.i + 1)

            running_rmse += rmse.item()
            cur_rmse = running_rmse.item() / (self.i + 1)

            desc = (f"[{self.epoch+1}] Train - DOA accuracy: {acc.item():.4f} / {cur_acc:.4f}, DOA RMSE: {rmse.item():.4f} / {cur_rmse:.4f}, "
                    f"Total Loss: {loss.item():.4f} / {cur_loss:.4f}, Regression Loss: {loss_m.item():.4f} / {cur_m_loss:.4f}, "
                    f"Reconstruction Loss: {loss_r.item():.4f} / {cur_r_loss:.4f}, Power Loss: {loss_p.item():.4f} / {cur_p_loss:.4f}")
            pbar.set_description(desc)
            pbar.update()

        self.optimizer_lr_sched.step()

        pbar.close()

        running_loss = running_loss.div_(len(self.train_loader))
        running_m_loss = running_m_loss.div_(len(self.train_loader))
        running_r_loss = running_r_loss.div_(len(self.train_loader))
        running_p_loss = running_p_loss.div_(len(self.train_loader))
        running_acc = running_acc.div_(len(self.train_loader))
        running_rmse = running_rmse.div_(len(self.train_loader))

        return running_loss.item(), running_m_loss.item(), running_r_loss.item(), running_p_loss.item(), running_acc.item(), running_rmse.item()
    
    def val_epoch(self):
        """
        """
        id = self.data_id()
        torch.save(self.model.state_dict(), self.model_path + "/deepbeamformer" + id + ".pt")

        pbar = tqdm(total=len(self.val_loader), position=0, leave=True)

        running_loss = torch.tensor(0., requires_grad=False).to(self.device)
        running_m_loss = torch.tensor(0., requires_grad=False).to(self.device)
        running_r_loss = torch.tensor(0., requires_grad=False).to(self.device)
        running_p_loss = torch.tensor(0., requires_grad=False).to(self.device)
        running_acc = torch.tensor(0., requires_grad=False).to(self.device)
        running_rmse = torch.tensor(0., requires_grad=False).to(self.device)

        with torch.no_grad():
            for self.i, (batch) in enumerate(self.val_loader):
            
                X, SCM, SOI, DOA, DOA_k_hot = batch
                X, SCM, SOI, DOA, DOA_k_hot = X.to(self.device), SCM.to(self.device), SOI.to(self.device), DOA.to(self.device), DOA_k_hot.to(self.device)
                        
                loss, loss_m, loss_r, loss_p, acc, rmse = self.loss_beamformer(X, SCM, SOI, DOA, DOA_k_hot, "val")

                running_loss += loss.item()
                cur_loss = running_loss.item() / (self.i + 1)

                running_m_loss += loss_m.item()
                cur_m_loss = running_m_loss.item() / (self.i + 1)

                running_r_loss += loss_r.item()
                cur_r_loss = running_r_loss.item() / (self.i + 1)

                running_p_loss += loss_p.item()
                cur_p_loss = running_p_loss.item() / (self.i + 1)
                
                running_acc += acc.item()
                cur_acc = running_acc.item() / (self.i + 1)

                running_rmse += rmse.item()
                cur_rmse = running_rmse.item() / (self.i + 1)

                desc = (f"[{self.epoch+1}] Validation - DOA accuracy: {acc.item():.4f} / {cur_acc:.4f}, DOA RMSE: {rmse.item():.4f} / {cur_rmse:.4f}, "
                        f"Total Loss: {loss.item():.4f} / {cur_loss:.4f}, Regression Loss: {loss_m.item():.4f} / {cur_m_loss:.4f}, "
                        f"Reconstruction Loss: {loss_r.item():.4f} / {cur_r_loss:.4f}, Power Loss: {loss_p.item():.4f} / {cur_p_loss:.4f}")
                pbar.set_description(desc)
                pbar.update()

            pbar.close()
            running_loss = running_loss.div_(len(self.val_loader))
            running_m_loss = running_m_loss.div_(len(self.val_loader))
            running_r_loss = running_r_loss.div_(len(self.val_loader))
            running_p_loss = running_p_loss.div_(len(self.val_loader))
            running_acc = running_acc.div_(len(self.val_loader))
            running_rmse = running_rmse.div_(len(self.val_loader))

        return running_loss.item(), running_m_loss.item(), running_r_loss.item(), running_p_loss.item(), running_acc.item(), running_rmse.item()
