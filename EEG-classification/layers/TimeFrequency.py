import torch
import torch.nn as nn


class TimeFrequency(nn.Module):
    def __init__(self,
                 input_example: torch.Tensor,
                 bands,
                 hop_length: int,
                 fs: int = 128,
    ):
        super().__init__()

        n_fft = input_example[0].shape[0]

        self.n_fft, self.hop_length = n_fft, hop_length
        freqs = torch.fft.rfftfreq(n_fft, 1/fs)
        mask = []
        for (low,high) in bands:
            mask.append((freqs>=low)&(freqs<high))
        self.register_buffer('band_mask', torch.stack(mask, 0))

    def forward(self, x: torch.Tensor):
        N, C, L = x.shape
        x = x.view(N*C, L)
        st = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                        window=torch.hann_window(self.n_fft),
                        return_complex=True)
        mag = st.abs()
        out = (mag.unsqueeze(1) * self.band_mask[:, :, None]).mean(2)
        out = out.view(N, C, -1, out.shape[-1])
        return out