import torch
import torch.nn as nn

bands = [
    (0.5,4),
    (4,8),
    (8,12),
    (12,30),
    (30,63.9)
]

class TimeFrequency(nn.Module):
    def __init__(self,
                 input_example: torch.Tensor,
                 fs: int = 128,
                 hop_length: int = 128
    ):
        super().__init__()

        n_fft = input_example[0].shape[0]

        # bands = []
        # deltafreq = fs / hop_length
        # nbbands = int((fs / 2) / deltafreq)

        # for i in range(nbbands):
        #     bands.append((i*deltafreq, (i+1)*deltafreq))

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