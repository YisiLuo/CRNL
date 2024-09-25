import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch import nn, optim 
import numpy as np 
import scipy.io as scio
import matplotlib.pyplot as plt 
import scipy.io
import math
from utils_patch import *

def main():
    
    # for color image:
    data = "data/peppers"
    c = "2"
    max_iter = 401
    omega_0_4D = 2
    down_2 = [1,1,1,6]
    lr_real = 0.0001
    gamma = 2*10e-6
    p = 6; k = 30;
    
    '''
    # for hyperspectral data
    max_iter = 15001
    omega_0_4D = 12
    down_2 = [1,1,3,5]    
    lr_real = 0.0002
    gamma = 10e-6
    p = 6; k = 20;
    
    # for video data
    max_iter = 3001
    omega_0_4D = 2
    down_2 = [1,1,3,5] 
    gamma = 10e-6
    lr_real = 0.0001
    p = 6; k = 20;
    '''
    
    file_name = data+'gt.mat'
    mat = scipy.io.loadmat(file_name)
    X_np = mat["Ohsi"][:,:,:]
    
    n1_real = X_np.shape[0]
    n2_real = X_np.shape[1]
    
    [n_1,n_2,n_3] = X_np.shape
    
    n_1_real = n_1
    n_2_real = n_2
    
    show = [0,1,2] # band
    
    file_name = data+'p'+c+'.mat'
    mat = scipy.io.loadmat(file_name)
    X_np = mat["Nhsi"][:,:,:]
    X = torch.from_numpy(X_np).type(dtype).cuda()
    [n_1,n_2,n_3] = X.shape
    
    file_name = data+'gt.mat'
    mat = scipy.io.loadmat(file_name)
    gt_np = mat["Ohsi"][:,:,:]
    
    file_name = data+'p'+c+'TCTV.mat'
    mat = scipy.io.loadmat(file_name)
    com_np = mat["clean_image"][:,:,:]
    com = torch.from_numpy(com_np).type(dtype).cuda()
    
    ps_com = psnr3d(gt_np, com_np)
    
    mask = torch.ones(X.shape).type(dtype)
    mask[X == 0] = 0 
    X[mask == 0] = 0
    
    X_Out = com.clone().detach()
    mask = mask.clone().detach()
    
    [X_Out_, mask] = add_pad(X_Out, mask, p)
    [X_Out_, X] = add_pad(X_Out, X, p)
    X_Out = X_Out_
    
    n_1_real = X_Out.shape[0]; n_2_real = X_Out.shape[1];
    
    [X_patch, mask_patch] = patch_assign(X_Out, mask, p) 
    # [T = (n1-p+1)*(n2-p+1), p, p, n3]
    T = int((X_Out.shape[0]-p+1)*(X_Out.shape[1]-p+1))
    
    [X_patch_key, mask_patch_key] = patch_assign_key(X_Out, mask, p)
    # [L = ((n1-p)/(p-1)+1)*((n2-p)/(p-1)+1), p, p, n3]
    L = int(((X_Out.shape[0]-p)/(p-1)+1)*((X_Out.shape[1]-p)/(p-1)+1))
    l1 = int((X_Out.shape[0]-p)/(p-1)+1)
    l2 = int((X_Out.shape[1]-p)/(p-1)+1)
    
    [X_new, mask_new] = search_KNN_4D(X_patch_key, X_patch, 
                                   mask_patch_key, mask_patch, k, p, T, L)
    
    # [L, p, p, n3, k + 1]
    [n_1, n_2, n_3, n_4] = [p, p, n_3, k + 1]
    
    mid_channel = 20*int(n_2)

    r_1 = int(n_1/down_2[0]) 
    r_2 = int(n_2/down_2[1])
    r_3 = int(n_3/down_2[2])
    r_4 = int(n_4/down_2[3])
    centre = torch.Tensor(L, r_1, r_2, r_3, r_4).type(dtype)
    
    stdv = 0.1 / math.sqrt(centre.size(0))
    centre.data.uniform_(-stdv, stdv)
    
    U_input = torch.from_numpy(np.array(range(1,n_1+1))).reshape(n_1,1).type(dtype)
    V_input = torch.from_numpy(np.array(range(1,n_2+1))).reshape(n_2,1).type(dtype)
    W_input = torch.from_numpy(np.array(range(1,n_3+1))).reshape(n_3,1).type(dtype)
    Y_input = torch.from_numpy(np.array(range(1,n_4+1))).reshape(n_4,1).type(dtype)
    
    class SineLayer_4D(nn.Module):
        def __init__(self, in_features, out_features, bias=True,
                     is_first=False, omega_0=omega_0_4D):
            super().__init__()
            self.omega_0 = omega_0
            self.is_first = is_first
            
            self.in_features = in_features
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            
            self.init_weights()
        
        def init_weights(self):
            with torch.no_grad():
                if self.is_first:
                    self.linear.weight.uniform_(-1 / self.in_features, 
                                                 1 / self.in_features)      
                else:
                    self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                                 np.sqrt(6 / self.in_features) / self.omega_0)
            
        def forward(self, input):
            return torch.sin(self.omega_0 * self.linear(input))
        
    class Network_4D(nn.Module):
         def __init__(self, r_1,r_2,r_3,r_4,mid_channel):
             super(Network_4D, self).__init__()
             
             self.U_net = nn.Sequential(SineLayer_4D(1, mid_channel, is_first=True),
                                        SineLayer_4D(mid_channel, mid_channel, is_first=True),
                                        nn.Linear(mid_channel, r_1))
             
             self.V_net = nn.Sequential(SineLayer_4D(1, mid_channel, is_first=True),
                                        SineLayer_4D(mid_channel, mid_channel, is_first=True),
                                        nn.Linear(mid_channel, r_2))
             
             self.W_net = nn.Sequential(SineLayer_4D(1, mid_channel, is_first=True),
                                        SineLayer_4D(mid_channel, mid_channel, is_first=True),
                                        nn.Linear(mid_channel, r_3))
             
             self.Y_net = nn.Sequential(SineLayer_4D(1, mid_channel, is_first=True),
                                        SineLayer_4D(mid_channel, mid_channel, is_first=True),
                                        nn.Linear(mid_channel, r_4))
             
         def forward(self, centre, U_input, V_input, W_input, Y_input):
             U = self.U_net(U_input)
             V = self.V_net(V_input)
             W = self.W_net(W_input)
             Y = self.Y_net(Y_input)
             
             centre = centre.permute(0,2,3,4,1) # L r2 r3 r4 r1
             centre = centre @ U.t()
             
             centre = centre.permute(0,4,2,3,1) # L n1 r3 r4 r2
             centre = centre @ V.t()
             
             centre = centre.permute(0,1,4,3,2) # L n1 n2 r4 r3
             centre = centre @ W.t()
             
             centre = centre.permute(0,1,2,4,3) # L n1 n2 n3 r4
             centre = centre @ Y.t()
             
             return centre
    
    model = Network_4D(r_1, r_2, r_3, r_4, mid_channel).type(dtype)
    
    params = []
    params += [x for x in model.parameters()]
    centre.requires_grad=True
    params += [centre]
    optimizier = optim.Adam(params, lr=lr_real)
    
    for iter in range(max_iter):
        
        X_Out_new = model(centre, U_input, V_input, W_input, Y_input)
        
        loss = torch.norm(X_Out_new*mask_new-X_new*mask_new,2)
        
        X_Out_real = patch_recover_4D_new(X_Out_new, p, n_1_real, 
                                          n_2_real, n_3, l1, l2)
        
        loss = loss + gamma * torch.norm(X_Out_real[1:,:,:]-X_Out_real[:-1,:,:], 1)
        loss = loss + gamma * torch.norm(X_Out_real[:,1:,:]-X_Out_real[:,:-1,:], 1)

        optimizier.zero_grad()
        loss.backward()
        optimizier.step()

        if iter % 100 == 0:
            X_Out_real[mask == 1] = X[mask == 1].clone()
            
            X_Out_real = X_Out_real[0:n1_real,0:n2_real,:].clone()
            ps = psnr3d(gt_np, X_Out_real.cpu().detach().numpy())
            
            print('iteration:',iter,'CRNL PSNR',ps,'TCTV PSNR:',ps_com)
            
            
            plt.figure(figsize=(15,45))
            
            plt.subplot(131)
            plt.imshow(np.clip(np.stack((X[:,:,show[0]].cpu().detach().numpy(),
                                 X[:,:,show[1]].cpu().detach().numpy(),
                                 X[:,:,show[2]].cpu().detach().numpy()),2),0,1))
            plt.title('observed')
    
            plt.subplot(132)
            plt.imshow(np.clip(np.stack((X_Out_real[:,:,show[0]].cpu().detach().numpy(),
                                 X_Out_real[:,:,show[1]].cpu().detach().numpy(),
                                 X_Out_real[:,:,show[2]].cpu().detach().numpy()),2),0,1))
            plt.title('CRNL')
            
            plt.subplot(133)
            plt.imshow(np.clip(np.stack((X_Out[:,:,show[0]].cpu().detach().numpy(),
                                 X_Out[:,:,show[1]].cpu().detach().numpy(),
                                 X_Out[:,:,show[2]].cpu().detach().numpy()),2),0,1))
            plt.title('TCTV')
            plt.show()
                
                
                
main()   