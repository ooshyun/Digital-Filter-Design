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

from slib.python.debug.log import maker_logger
_logger = maker_logger()

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
    return [sum((h[k] * iexp(-2 * math.pi * i * k / n) for k in range(n)))
            for i in range(n)]


def dftinv(h):
    """
        naive dft(0 ~ N)
    """
    n = len(h)
    return [sum((h[k] * iexp(2 * math.pi * i * k / n) for k in range(n))) / n
            for i in range(n)]

if __name__ == "__main__":
    total_sample = samplingFreq

    # t = np.arange(-1 * np.round(total_sample / 2) + 1, np.round(total_sample / 2) + 1)
    t = np.arange(0, N)
    Hm = np.zeros(N)

    n1 = np.arange(-N/2+1, N/2+1)

    Hm1 = Hm.copy()

    index_origin = np.where(n1==0)[0][0]

    Hm1[index_origin:index_origin+int(K/2)+1] = 1
    Hm1[index_origin-int(K/2):index_origin] = 1
    Hm1 = np.roll(Hm1, int(N/2)+1)

    hk = dftinv(Hm1)

    """
    Tap
        * Upper this line is explained in fir_freq2time.py

        How many bandwidth you want ?

        Compare 7, 19, 31 tap

        h(Small character) means time domain filter
        H(Large character) means freq domain filter
    """
    def tap(number_tap, transfer):
        """
         Get specific number of coefficients from the time domian transfer function 
        """
        if number_tap%2 == 0:
            return
        else:
            h = np.zeros(len(transfer), dtype=np.complex128)
            h[0:int(number_tap/2)+1] = transfer[0:int(number_tap/2)+1]
            h[-int(number_tap/2):] = transfer[-int(number_tap/2):]
            
            return dft(h)

    # Each tap for filter
    h_tap_7 = tap(7, hk)
    h_tap_19 = tap(19, hk)
    h_tap_31 = tap(31, hk)

    # Check the tap form
    # plt.plot(np.abs(np.array(h_tap_7, dtype=np.complex128)))
    # plt.show()

    """ Comparision between Square 31-tap, 19-tap, 7-tap """
    fig = plt.figure(constrained_layout=True, figsize=(24, 8))
    axes = fig.subplots(ncols=1, nrows=3)

    axes[0].plot(h_tap_7, 'b.')
    # axes[0].legend(loc='lower center')

    axes[1].plot(h_tap_19, 'r.', label = '19-tap')
    axes[1].legend(loc='lower center')

    axes[2].plot(h_tap_31, 'k.', label = '31-tap')
    axes[2].legend(loc='lower center')
    plt.show()