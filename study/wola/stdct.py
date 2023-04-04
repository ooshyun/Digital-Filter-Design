import IPython.display as ipd
import librosa
import librosa.display
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import resampy 
import scipy.fft as scfft

sample_rate = 22050
wav, sr = librosa.load('./study/wola/data/14-208-0005_3347-134425-0010.wav', sr=sample_rate)
# wav = resampy.resample(wav, sr_orig=sr, sr_new=22050)


nfft = 1024
win_length = nfft
hop_length = nfft//4 

name_window = 'han' #'square'  'triang', 'han', 'hamming'

if name_window == 'square':
    window = np.ones(shape=(win_length,))
else:
    window = signal.get_window(window=name_window, Nx=nfft)

# print(np.max(window), np.min(window))

wav_padded = wav.astype(np.float32)

# print(wav_padded.shape)

padding = [(0, 0) for _ in range(wav_padded.ndim)]
# padding[-1] = (3*int(nfft // 4), int(nfft // 2))
padding[-1] = (int(nfft // 2), int(nfft // 2))
wav_padded = np.pad(wav_padded, padding, mode="constant")

# print(wav_padded.shape)

num_frame = int((wav_padded.shape[-1] - nfft ) //hop_length)+1
frames = np.zeros(shape=(num_frame, nfft), dtype=np.float32)

curr_index = 0
for n_frame in range(num_frame):
    frames[n_frame, ...] = window*wav_padded[...,curr_index:curr_index+nfft]
    curr_index += hop_length

frames = frames.T

# fft
fft_frames = np.fft.fft(frames, n=nfft, axis=-2)
print(fft_frames.shape, fft_frames.dtype)

fft_frames_stft = librosa.stft(y=wav, n_fft=nfft, hop_length=hop_length, 
                            win_length=nfft, window=name_window, center=True,
                            dtype=np.complex128, pad_mode="constant")
print(fft_frames_stft.shape)  


dct_frames = scfft.dct(frames, type=2, n=nfft, axis=-2)

spectogram_librosa = librosa.amplitude_to_db(np.abs(fft_frames_stft))
spectogram_fft_frames = librosa.amplitude_to_db(np.abs(fft_frames[...,:int(nfft//2+1), :]))
spectogram_dct_frames = librosa.amplitude_to_db(np.abs(dct_frames[...,:int(nfft//2+1), :]))


fig, ax = plt.subplots(nrows=2)
img = librosa.display.specshow(spectogram_librosa, y_axis='linear', sr=sr, hop_length=hop_length,
                          x_axis='time', ax=ax[0])
librosa.display.specshow(spectogram_dct_frames, y_axis='linear', sr=sr, hop_length=hop_length,
                          x_axis='time', ax=ax[1])
ax[0].set(title='DFT Log-frequency power spectrogram')
ax[0].label_outer()

ax[1].set(title='DCT Log-frequency power spectrogram')
ax[1].label_outer()

fig.colorbar(img, ax=ax, format="%+2.f dB")

plt.show()