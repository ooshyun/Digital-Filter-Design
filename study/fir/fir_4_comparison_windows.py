"""
    Book "Understanding Digital Signal Processing. Ch 5. 181 page
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.core.fromnumeric import argmax
from scipy.fft import fft, ifft
from scipy.io.wavfile import write
import scipy.signal
# from scipy.signal import get_window

samplingFreq = 16

bandwidth = 8
coeff = bandwidth * 2 + 1

K = 7

N = samplingFreq
j = complex(0, 1)

x = np.arange(10)


# DFT
def is_pow2(n):
    return False if n == 0 else (n == 1 or is_pow2(n >> 1))


def iexp(n):
    return np.complex128(complex(math.cos(n), math.sin(n)))


def dft(h):
    """naive dft"""
    n = len(h)
    return [sum((h[k] * iexp(-2 * math.pi * i * k / n) for k in range(n)))
            for i in range(n)]
                        

def dftinv(h):
    """
        naive idft(0 ~ N)
    """
    n = len(h)
    return [sum((h[k] * iexp(2 * math.pi * i * k / n) for k in range(n))) / n
            for i in range(n)]


def blackmanWindow_naive(size:int):
    return np.array(
        [0.42 - 0.5 * np.cos(2 * np.pi * k / (size - 1)) + 0.08 * np.cos(4 * np.pi * k / (size - 1)) for k in
         range(size)])

def blackmanWindow(size:int, sym:bool=False):
    return scipy.signal.windows.blackman(size, sym=sym)

def chebyshevWindow(size:int, sym:bool=False):
    return scipy.signal.windows.chebwin(size, at=45, sym=sym)


def kaiserWindow(size:int, beta:float=4, sym:bool=False):
    return scipy.signal.windows.kaiser(size, beta, sym=sym)

if __name__ == "__main__":
    """
        For Checking the window frequency response
    """

    window_dict = {}
    rectangle = np.ones(N)/N
    
    # samplingFreq = N
    window_dict['Rectangle'] = rectangle
    window_dict['Blackman Window'] = rectangle*blackmanWindow(N, sym=False)
    window_dict['Blackman Window_naive'] = rectangle*blackmanWindow_naive(N)
    window_dict['Chebyshev Window'] = rectangle*chebyshevWindow(N, sym=False)
    window_dict['Kaiser Window'] = rectangle*kaiserWindow(N, sym=False)

    def plot_all(datas: list):
        for data in datas:
            print(data)
            plt.plot(data, ".")
        plt.grid()
        plt.show()

    # Check the alignment    
    # sw(list(window_dict.values()))
     
    def zero_padding(h, n: int):
        assert len(h)%2==0

        if len(h) >= n:
            return h
        else:
            transfer = h.copy()
            n_dft = n-len(h)
            zero_padding = np.zeros((n_dft)//2)
            # bias = transfer[0]
            # transfer = transfer[1:]
            transfer = np.append(zero_padding, transfer)
            transfer = np.append(transfer, zero_padding)
            # transfer = np.append(bias, transfer)
            return transfer

    for label, data in zip(window_dict.keys(), window_dict.values()):
        buf = zero_padding(data, n=512)
        buf = np.roll(buf, len(buf)//2)
        window_dict[label] = dft(buf)

    del buf

    plot_all(list(window_dict.values()))