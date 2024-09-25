import torch
from skimage.metrics import peak_signal_noise_ratio

dtype = torch.cuda.FloatTensor

def psnr3d(x,y):
    p=0
    for i in range(x.shape[2]):
        ps = peak_signal_noise_ratio(x[:,:,i],y[:,:,i])
        p = p + ps
        # print(ps)
    return p/x.shape[2]

def patch_assign(X_Out, mask, p):
    print('Patch constructing...')
    n1 = X_Out.shape[0]-p+1
    n2 = X_Out.shape[1]-p+1
    X_patch = torch.zeros((n1)*(n2), p, p, X_Out.shape[2]).type(dtype)
    mask_patch = torch.zeros((n1)*(n2), p, p, X_Out.shape[2]).type(dtype)
    for i in range(n1):
        for j in range(n2):
            X_patch[(i*n2)+j, :, :, :] = X_Out[i:i+p, j:j+p, :].clone().detach()
            mask_patch[(i*n2)+j, :, :, :] = mask[i:i+p, j:j+p, :].clone().detach()
    return X_patch, mask_patch 
    # [T = (n1-p+1)*(n2-p+1), p, p, n3]
    
def add_pad(X_Out, mask, p):
    
    n1 = X_Out.shape[0]
    n2 = X_Out.shape[1]
    
    if (n1 - p) % ((p-1)) == 0 and (n2 - p) % ((p-1)) == 0:
        return X_Out, mask 
    
    res_1 = p - 1 - (n1 - p) % ((p-1))
    res_2 = p - 1 - (n2 - p) % ((p-1))
    
    X_Out_1 = torch.zeros(n1+res_1,
                          n2+res_2,
                          X_Out.shape[2]).type(dtype)
    
    mask_1 = torch.zeros(n1+res_1,
                          n2+res_2,
                          mask.shape[2]).type(dtype)
    
    X_Out_1[0:n1, 0:n2, :] = X_Out.clone()
    mask_1[0:n1, 0:n2, :] = mask.clone()
    
    X_Out_1[n1:, :n2, :] = X_Out[n1-res_1:, :, :].clone()
    X_Out_1[:n1, n2:, :] = X_Out[:, n2-res_2:, :].clone()
    
    mask_1[n1:, :n2, :] = mask[n1-res_1:, :, :].clone()
    mask_1[:n1, n2:, :] = mask[:, n2-res_2:, :].clone()
    
    X_Out_1[n1:, n2:, :] = X_Out[n1-res_1:, n2-res_2:, :].clone()
    mask_1[n1:, n2:, :] = mask[n1-res_1:, n2-res_2:, :].clone()
    
    return X_Out_1, mask_1
    

def patch_assign_key(X_Out, mask, p):
    print('Patch key constructing...')
    n1 = int((X_Out.shape[0]-p)/(p-1)+1)
    n2 = int((X_Out.shape[1]-p)/(p-1)+1)
    X_patch_key = torch.zeros((n1)*(n2), p, p, X_Out.shape[2]).type(dtype)
    mask_patch_key = torch.zeros((n1)*(n2), p, p, X_Out.shape[2]).type(dtype)
    for i in range(n1):
        for j in range(n2):
            X_patch_key[(i*n2)+j, :, :, :] = X_Out[i*(p-1):i*(p-1)+p, 
                                                   j*(p-1):j*(p-1)+p, :].clone().detach()
            mask_patch_key[(i*n2)+j, :, :, :] = mask[i*(p-1):i*(p-1)+p,
                                                     j*(p-1):j*(p-1)+p, :].clone().detach()
    
    return X_patch_key, mask_patch_key
    # [L = ((n1-p)/(p-1)+1)*((n2-p)/(p-1)+1), p, p, n3]


def search_KNN_4D(X_patch_key, X_patch, mask_patch_key, mask_patch, k, p, T, L):
    print("Searching KNN...")
    X_new = torch.zeros(L, k+1, p, p, X_patch_key.shape[3]).type(dtype)
    mask_new = torch.zeros(L, k+1, p, p, X_patch_key.shape[3]).type(dtype)
    # [L, k+1, p, p, n3]
    
    X_patch_here = torch.flatten(X_patch.clone().detach(), start_dim=1)
    # [T, p*p*n3]
    
    for i in range(L):
        print('\r','Searching ', i,' key patch',end='')
        
        patch_key_here = torch.flatten(X_patch_key[i,:].clone().detach(),start_dim=0)
        # [p*p*n3]
        
        patch_key_here = patch_key_here.repeat(T, 1)
        # [T, p*p*n3]
        
        distance_list = torch.sum((patch_key_here - X_patch_here)**2, dim = 1)
        #distance_list = distance_list.numpy().tolist()
        # [T]
        
        X_new[i, 0, :, :, :] = X_patch_key[i, :, :, :].clone()
        mask_new[i, 0, :, :, :] = mask_patch_key[i, :, :, :].clone()
        
        _, inds = distance_list.topk(k, dim=0, largest = False)
            
        X_new[i, 1:, :, :, :] = X_patch[inds, :, :, :].clone()
        mask_new[i, 1:, :, :, :] = mask_patch[inds, :, :, :].clone()
        
    X_new = X_new.permute(0, 2, 3, 4, 1)
    mask_new = mask_new.permute(0, 2, 3, 4, 1)
    
    return X_new, mask_new
    # [L, p, p, n_3, k+1]

def patch_recover_4D_new(X_Out_new, p, n1, n2, n3, l1, l2):
    
    # [L, p, p, n_3, k+1] -> [n1, n2, n3]
    X_Out = (X_Out_new[:,:,:,:,0]).reshape(l1,l2,p,p,n3)
    X_Out = X_Out.permute(0,2,1,3,4).reshape(p*l1, p*l2, n3)
    
    T = X_Out.clone().detach()
    
    T[p-1:int(p*(n1-p)/(p-1)+1):p,:,:] = 10e5
    T[:,p-1:int(p*(n2-p)/(p-1)+1):p,:] = 10e5
    mask = torch.ones(X_Out.shape, dtype=torch.bool).cuda()
    
    mask[T == 10e5] = False
    mask[T != 10e5] = True
    
    X_Out_ = torch.masked_select(X_Out, mask)

    return X_Out_.reshape(n1,n2,n3)