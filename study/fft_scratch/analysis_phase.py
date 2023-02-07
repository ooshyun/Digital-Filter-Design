import librosa
import numpy as np
import matplotlib.pyplot as plt

f = 1000
phi = np.pi

t = np.arange(1024)
fs = 16000
mag = 1

signal_1 = mag*np.sin(2*np.pi*f*t/fs)
signal_2 = mag*np.sin(2*np.pi*f*t/fs + phi)

plt.plot(signal_1)
plt.plot(signal_2)
plt.show()


# signal_1, sr = librosa.load(librosa.ex("trumpet"))[:1024]

def amp_phase(s):
    fft = np.fft.fft(s, 1024)
    mag_fft = np.abs(fft[:len(fft)//2+1])
    phase_fft = np.angle(fft[:len(fft)//2+1])/(2*np.pi)
    return mag_fft, phase_fft

mag_fft_1, phase_fft_1 = amp_phase(signal_1)
mag_fft_2, phase_fft_2 = amp_phase(signal_2)

print(mag_fft_1[64]) # 512
print(mag_fft_2[64]) # 512
print(phase_fft_1[64]) # -0.25
print(phase_fft_2[64]) #  0.25

# fig, (ax0, ax1) = plt.subplots(nrows=2)
# ax0.plot(mag_fft_1, ".")
# ax0.plot(mag_fft_2, "x")

# ax1.plot(phase_fft_1, ".")
# ax1.plot(phase_fft_2, "*")

# plt.show()