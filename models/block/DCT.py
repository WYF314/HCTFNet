import torch
import torch.nn as nn
import numpy as np

class DCT(nn.Module):
    def __init__(self, high_freq_weight=2.0, low_freq_weight=1.0):
        super().__init__()
        self.high_w = high_freq_weight
        self.low_w = low_freq_weight

    def _dct_matrix(self, n):
        dct_m = np.zeros((n, n), dtype=np.float32)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    dct_m[k, i] = 1 / np.sqrt(n)
                else:
                    dct_m[k, i] = np.sqrt(2/n) * np.cos(np.pi * k * (2*i + 1) / (2*n))
        return torch.tensor(dct_m)

    def dct_2d(self, x):
        B, C, H, W = x.shape
        dct_h = self._dct_matrix(H).to(x.device)
        dct_w = self._dct_matrix(W).to(x.device)
        dct = torch.matmul(torch.matmul(dct_h, x), dct_w.t())
        return dct

    def idct_2d(self, x):
        B, C, H, W = x.shape
        dct_h = self._dct_matrix(H).to(x.device)
        dct_w = self._dct_matrix(W).to(x.device)
        idct = torch.matmul(torch.matmul(dct_h.t(), x), dct_w)
        return idct

    def forward(self, x):
        B, C, H, W = x.shape
        
        dct = self.dct_2d(x)

        h_split = H // 2
        w_split = W // 2
        dct_enhanced = dct.clone()
        dct_enhanced[:, :h_split, :w_split] = dct[:, :h_split, :w_split] * self.low_w
        dct_enhanced[:, h_split:, :] = dct[:, h_split:, :] * self.high_w
        dct_enhanced[:, :, w_split:] = dct[:, :, w_split:] * self.high_w
        dct_enhanced[:, h_split:, w_split:] = dct[:, h_split:, w_split:] * self.high_w
        
        idct = self.idct_2d(dct_enhanced)
        
        return idct