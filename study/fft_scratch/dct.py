
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct

# TODO
def dct_scratch(f, type=1, M=0):
    """
        DCT-1, T = 2M-2
        Fk  = sigma_{m=0}^{2M-3}*f_m*e^{-j*pi*k/(M-1)*m}
            = f_0 + (-1)^k*f_{M-1} + 2*sigma_{m=1}^{M-2}*f_m*cos(pi*k*(m-1)/(M-1)*m)
        
        DCT-1, Xk = Fk/2

        DCT-2, T = 4M
        F_k = 2*sigma_{m=0}^{M-1}*f_{2m+1}*cos(pi*k/2M*m)
    """
    if M == 0:
        M = len(f)    
    
    fft_result = np.zeros(M, dtype=np.float64)
    if type == 1:
        for k in range(len(fft_result)):
            summation_f = 0
            summation_f += (f[0]+f[-1]*(-1)**k)/2  # (x_0+(-1)^{-k}*x_{N-1})/2
            
            for m in range(1, len(f)-1):
                summation_f += 2*f[m]* np.cos((np.pi*k*m)/M)

            fft_result[k] = summation_f

    elif type == 2:
        for k in range(len(fft_result)):
            summation_f = 0
            
            for m in range(1, len(f)-1):
                summation_f += 2*f[m]*np.cos(np.pi*k*(2*m+1)/(2*(M+1)))
            
            fft_result[k] = summation_f

    return fft_result

# TODO
def idct_scratch(f, M=0):
    """
        iDCT-1
        x_m = 1/(M-1) * [X_0 + (-1)^m*X_{M-1} + 2*sigma_{m=1}^{M-2}*X_m*cos(pi*m*(m-1)/(M-1)*m)] ]
    """
    raise NotImplementedError


# TODO
def dst_scratch(f, M=0):
    raise NotImplementedError

# TODO
def idst_scratch(f, M=0):
    raise NotImplementedError


if __name__=="__main__":
    print("Checking function operation...")
    print("-" * 40)    

    # x = np.random.random(1024)
    # print("DCT-|")
    # print(np.allclose(dct(x, type=1), dct_scratch(x, type=1), atol=1e-2))

    # print("DCT-||")
    # print(np.allclose(dct(x, type=2), dct_scratch(x, type=2), atol=1e-2))
        

    # print("Check function operation in frequency domain...")

    # fs = 1000
    # total_time = 0.1
    # frequency = 100
    # sample_list = np.arange(0, total_time * fs)
    # a_0 = np.sin(2 * np.pi * frequency * sample_list / fs)
    # a_1 = np.sin(2 * np.pi * 2 * frequency * sample_list / fs)
    # # a = (a_0 + a_1) / 2
    # a = a_0

    # n = 2048
    # bins = np.arange(0, n) * fs / n

    # a_fft_numpy = np.fft.fft(a, n=n)
    # a_dct_scipy_type_1 = dct(a, type=1, n=n)
    # a_idct_scipy_type_1 = idct(a_dct_scipy_type_1, type=1, n=n)
    # a_dct_scratch_type_1 = dct_scratch(a, type=1, M=n)


    # a_dct_scipy_type_2 = dct(a, type=2, n=n)
    # a_idct_scipy_type_2 = idct(a_dct_scipy_type_2, type=1, n=n)
    # a_dct_scratch_type_2 = dct_scratch(a, type=2, M=n)

    # fig, (ax0, ax1) = plt.subplots(nrows=2)
    # ax0.plot(a)

    # ax1.plot(bins, np.abs(a_fft_numpy), "o")
    # ax1.plot(bins, np.abs(a_dct_scipy_type_1), "x")
    # ax1.plot(bins, np.abs(a_dct_scratch_type_1), ".")

    # ax1.plot(bins, np.abs(a_dct_scipy_type_2), "x")
    # ax1.plot(bins, np.abs(a_dct_scratch_type_2), ".")
    
    # plt.show()

    print("Check function operation in frequency domain 2...")

    import resampy
    import librosa

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4)
    f = 1000
    sample_rate = 16000

    # https://librosa.org/doc/main/recordings.html
    # signal, sr = librosa.load(librosa.ex("libri1"), sr=sample_rate)[:sample_rate]
    # signal, sr = librosa.load(librosa.ex("trumpet"), sr=sample_rate)[:sample_rate]
    # signal, sr = librosa.load(librosa.ex("sweetwaltz"), sr=sample_rate)[:sample_rate]
    signal, sr = librosa.load(librosa.ex("fishin"), sr=sample_rate)[:sample_rate]
    signal_double_sr = resampy.resample(signal, sample_rate, sample_rate*2)[:sample_rate]
    signal_half_sr = resampy.resample(signal, sample_rate, sample_rate//2)[:sample_rate]

    # signal_double_sr = np.sin(2*np.pi*f*np.arange(16000)/(2*sample_rate))
    # signal = np.sin(2*np.pi*f*np.arange(16000)/sample_rate)
    # signal_half_sr = np.sin(2*np.pi*f*np.arange(16000)/(sample_rate//2))

    fft_1k_double_sr = np.fft.fft(signal_double_sr, n=sample_rate)
    fft_1k = np.fft.fft(signal, n=sample_rate)
    fft_1k_half_sr = np.fft.fft(signal_half_sr, n=sample_rate)
    dct_1k = dct(signal, type=2, n=sample_rate)


    amp_fft_1k_double_sr = np.abs(fft_1k_double_sr)
    amp_fft_1k = np.abs(fft_1k)
    amp_fft_1k_half_sr = np.abs(fft_1k_half_sr)
    amp_dct_1k = np.abs(dct_1k)
    

    ax0.plot(np.arange(sample_rate)*2*sample_rate/sample_rate, amp_fft_1k_double_sr, label="dft double sr")
    ax1.plot(np.arange(sample_rate)*sample_rate/sample_rate, amp_fft_1k, label="dft sr")
    ax2.plot(amp_dct_1k, label="dct")
    ax3.plot(np.arange(sample_rate)*sample_rate//2/sample_rate, amp_fft_1k_half_sr, label="dft half sr")

    ax0.grid()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    plt.show()

    ifft_1k = np.fft.ifft(fft_1k, n=sample_rate).real
    idct_1k = idct(dct_1k, type=2, n=sample_rate)

    print(np.max(np.abs(ifft_1k-idct_1k)))
    print(np.max(np.abs(ifft_1k-signal)))
    print(np.max(np.abs(idct_1k-signal)))
    
    # plt.plot(ifft_1k)
    # plt.plot(idct_1k)
    # plt.plot(signal)

    # plt.grid()
    # plt.show()