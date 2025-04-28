import torch
import torch.nn as nn

class TimeFrequency(nn.Module):
    def __init__(self,
                 fs: int = 128,
                 n_fft: int = 512,
                 hop_length: int = 128,
                 bands: dict = {
                     'delta': (0.5,4),
                     'theta': (4,8),
                     'alpha': (8,12),
                     'beta':  (12,30),
                     'gamma': (30,63.9)
                 }):
        super().__init__()
        self.fs, self.n_fft, self.hop_length = fs, n_fft, hop_length
        # build band‐mask tensor of shape (5, n_fft//2+1)
        freqs = torch.fft.rfftfreq(n_fft, 1/fs)  # (F,)
        mask = []
        for (low,high) in bands.values():
            mask.append((freqs>=low)&(freqs<high))
        self.register_buffer('band_mask', torch.stack(mask, 0))  # (5, F)

    def forward(self, x: torch.Tensor):
        # x: (N, C, L)
        N, C, L = x.shape
        x = x.view(N*C, L)
        st = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                        window=torch.hann_window(self.n_fft),
                        return_complex=True)       # (N*C, F, T)
        mag = st.abs()                        # (N*C, F, T)
        # for each band, mean over its freq-bins → (N*C,5,T)
        out = (mag.unsqueeze(1) * self.band_mask[:, :, None]).mean(2)
        out = out.view(N, C, -1, out.shape[-1])  # (N, C, 5, T)
        return out