import torch
import librosa
import matplotlib.pyplot as plt

x, sr = librosa.load(librosa.ex("trumpet"))
x = x[:10000]

# plt.plot(x)
# plt.show()

x = torch.tensor(data=x, dtype=torch.float32)

n_fft = 512
hop_length = 256
win_length = n_fft

x_stft_non_normalized = torch.stft(input=x,
             n_fft=n_fft,
             hop_length=hop_length,
             win_length=win_length,
             normalized=False,
             return_complex=True)

# If :attr:`normalized` is ``True`` (default is ``False``), the function
# returns the normalized STFT results, i.e., multiplied by :math:`(\text{frame\_length})^{-0.5}`.

x_stft_normalized = torch.stft(input=x,
             n_fft=n_fft,
             hop_length=hop_length,
             win_length=win_length,
             normalized=True,
             return_complex=True)


x_stft_non_normalized = x_stft_non_normalized.abs().flatten()
x_stft_normalized = x_stft_normalized.abs().flatten()

fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.plot(x_stft_non_normalized)
ax1.plot(x_stft_normalized)

plt.show()
