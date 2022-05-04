"""
    Reference
    https://angeloyeo.github.io/2020/10/21/Savitzky_Golay.html
"""

import numpy as np
import matplotlib.pyplot as plt

M = 20  # filter length 2M+1 = 41
N = 9  # coeff order 9


# % 테스트용 신호
# load mtlb
# t = (0:length(mtlb)-1)/Fs;
import scipy.io.wavfile

path = (
    "/Users/seunghyunoh/workplace/Data/SpeechSample/Sample_CS101_UIC/CantinaBand3.wav"
)
wav = scipy.io.wavfile.read(path)
print(wav)
fs = wav[0]
import math

mtlb = wav[1] / (2 ** 16)
t = np.arange(0, len(mtlb)) / fs
import scipy.signal

# Case 1
b = scipy.signal.savgol_coeffs(2 * M + 1, N)
sgolay_filter = b[int((len(b) + 1) / 2) :]
smtlb = scipy.signal.convolve(mtlb, b, "same")

# Case 2
smtlb_func = scipy.signal.savgol_filter(mtlb, 2 * M + 1, N)

# Case 3
A = np.zeros((2 * M + 1, N + 1))
n_range = np.arange(-M, M + 1)
i_range = np.arange(0, N + 1)

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        A[i][j] = n_range[i] ** i_range[j]

A_T = np.transpose(A)
H = np.dot(np.linalg.inv(np.dot(A_T, A)), A_T)
sgolay_filter_calculated = H[0, :]

my_smtlb_calculated = np.convolve(mtlb, sgolay_filter_calculated, "same")

h1 = plt.plot(t, mtlb, label="Origianl")
h2 = plt.plot(t, my_smtlb_calculated, "r", label="sgolay_filter_calculated")

# filter = plt.plot(sgolay_filter_calculated)
plt.legend()
plt.show()

# save to result folder
# save_path = "./result/sgfilter/"
# scipy.io.wavfile.write(save_path+"origin.wav",fs,mtlb)
# scipy.io.wavfile.write(save_path+"sgv.wav",fs, my_smtlb_calculated)
