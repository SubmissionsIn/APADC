import torch


def guassian_kernel_mmd(source, target, kernel_mul=2, kernel_num=4, fix_sigma=None):
    """Gram kernel matrix
    source: sample_size_1 * feature_size
    target: sample_size_2 * feature_size
    kernel_mul: bandwith of kernels
    kernel_num: number of kernels
    return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)
            [ K_ss K_st
              K_ts K_tt ]
    """
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) 

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val) 


def MMD(source, target, kernel_mul=2, kernel_num=4, fix_sigma=None):
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel_mmd(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n] 
    YY = kernels[n:, n:]
    XY = kernels[:n, m:]
    YX = kernels[m:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)   # K_ss，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1)  # K_st，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1)  # K_ts,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)   # K_tt,Target<->Target
    	
    loss = XX.sum() + XY.sum() + YX.sum() + YY.sum()                
    return loss

