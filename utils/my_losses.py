import torch

def frequency_loss(pred, true, dim=1):
    """
    计算基于rFFT的频域L1损失
    pred, true: [B, L, D]
    """
    pred_ft = torch.fft.rfft(pred, dim=dim)
    true_ft = torch.fft.rfft(true, dim=dim)
    return (pred_ft - true_ft).abs().mean()