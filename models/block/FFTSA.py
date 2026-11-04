import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialFrequencyAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, keep_freq='high', freq_ratio=0.3, alpha_init=1.0):
        super(SpatialFrequencyAttention, self).__init__()
        
        assert kernel_size % 2 == 1, "卷积核尺寸必须为3/5/7）"
        assert keep_freq in ['low', 'high'], "keep_freq仅支持'low'（低频）/'high'（高频）"
        assert 0 < freq_ratio < 1, "freq_ratio需在0~1之间"
        
        self.keep_freq = keep_freq
        self.freq_ratio = freq_ratio
        self.padding = kernel_size // 2
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.freq_conv = nn.Conv2d(
            in_channels=2,          
            out_channels=1,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=False              
        )
        self.sigmoid = nn.Sigmoid()
        
        nn.init.kaiming_normal_(self.freq_conv.weight, mode='fan_out', nonlinearity='relu')

    def _fft_freq_filter(self, x):
        b, c, h, w = x.shape
        # FFT
        fft_complex = torch.fft.fft2(x, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft_complex)
        freq_mask = torch.zeros_like(fft_shift)
        h_keep = int(h * self.freq_ratio)
        w_keep = int(w * self.freq_ratio)
        h_start = (h - h_keep) // 2
        h_end = h_start + h_keep
        w_start = (w - w_keep) // 2
        w_end = w_start + w_keep
        
        if self.keep_freq == 'low':
            freq_mask[:, :, h_start:h_end, w_start:w_end] = 1.0
        else:
            freq_mask[:, :, h_start:h_end, w_start:w_end] = 0.0
            freq_mask = 1.0 - freq_mask
        # IFFT
        fft_filtered = fft_shift * freq_mask
        # fft_ishift = torch.fft.ifftshift(fft_filtered)
        # x_filtered = torch.fft.ifft2(fft_ishift, dim=(-2, -1)).real
        
        return fft_filtered

    def forward(self, x):
        b, c, h, w = x.shape
        
        # high freq
        x_filtered = self._fft_freq_filter(x)
        
        # fft_complex = torch.fft.fft2(x_filtered, dim=(-2, -1))
        amplitude = torch.abs(x_filtered)
        phase = torch.angle(x_filtered)

        amplitude_avg = torch.mean(amplitude, dim=1, keepdim=True) # (B,1,H,W)
        amplitude_max, _ = torch.max(x, dim=1, keepdim=True)
        # feat = torch.cat([amplitude_avg, amplitude_max], dim=1)   


        freq_feat = torch.cat([amplitude_avg, amplitude_max], dim=1)   # (B,2,H,W)
        spatial_att = self.freq_conv(freq_feat)
        spatial_att = self.sigmoid(spatial_att)

        out = x + self.alpha * (x * spatial_att)
        
        return out, spatial_att