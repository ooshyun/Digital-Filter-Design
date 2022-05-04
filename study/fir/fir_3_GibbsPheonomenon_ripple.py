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

samplingFreq = 32

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
    return [
        sum((h[k] * iexp(-2 * math.pi * i * k / n) for k in range(n))) for i in range(n)
    ]


def dftinv(h):
    """
        naive dft(0 ~ N)
    """
    n = len(h)
    return [
        sum((h[k] * iexp(2 * math.pi * i * k / n) for k in range(n))) / n
        for i in range(n)
    ]


def blackmanWindow(size: int, sym: bool = False):
    # return np.array(
    #     [0.42 - 0.5 * np.cos(2 * np.pi * k / (size - 1)) + 0.08 * np.cos(4 * np.pi * k / (size - 1)) for k in
    #      range(size)])
    return scipy.signal.windows.blackman(size, sym=sym)


if __name__ == "__main__":
    total_sample = samplingFreq

    # t = np.arange(-1 * np.round(total_sample / 2) + 1, np.round(total_sample / 2) + 1)
    t = np.arange(0, N)
    Hm = np.zeros(N)

    n1 = np.arange(-N / 2 + 1, N / 2 + 1)

    denominate = np.sin(np.pi * t / N)
    nominate = np.sin((bandwidth * 2 + 1) * np.pi * t / N)
    Hsin = np.zeros(len(t))
    for k, term in enumerate(nominate):
        if nominate[k] == 0 and denominate[k] == 0:
            Hsin[k] = (bandwidth * 2 + 1) / N
        else:
            Hsin[k] = nominate[k] / denominate[k] / N

    hsin = dftinv(Hsin)

    def tap(number_tap, transfer):
        """
        Get specific number of coefficients from the time domian transfer function 
        """
        if number_tap % 2 == 0:
            return
        else:
            h = np.zeros(len(transfer), dtype=np.complex128)
            h[0 : int(number_tap / 2) + 1] = transfer[0 : int(number_tap / 2) + 1]
            h[-int(number_tap / 2) :] = transfer[-int(number_tap / 2) :]

            return dft(h)

    """
    Using the Window, page.186
    
        - Ripple known as Gibbs's pheonomenon
    """

    """
    coefficient는 짝수? 홀수?
        - 전체 길이는 짝수가 맞다!
    
    Example.
        16 sampling 
        -------------------------------------
        0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
        |               |  
        0   : bias
        1-7 : refleaction
        8   : fs/2
        9-15: refleaction
    """

    _transfer = hsin.copy()

    # 31-tap LPF
    h_nowindow = tap(31, _transfer)

    # Shift for applying Windows
    h_nowindow = np.roll(h_nowindow, len(h_nowindow) // 2)
    w = blackmanWindow(len(h_nowindow), sym=False)

    def plot_all(datas: list):
        for data in datas:
            plt.plot(data, ".")
        plt.grid()
        plt.show()

    # Check the alignments
    # plot_all([w, h_nowindow])

    # You can see the compressing the ripple after apply the window
    plot_all([w, h_nowindow, w * h_nowindow])
