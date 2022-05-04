"""
    Book "Understanding Digital Signal Processing. Ch 5. 181 page
"""
import numpy as np
import matplotlib.pyplot as plt
import math

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


if __name__ == "__main__":
    total_sample = samplingFreq

    # t = np.arange(-1 * np.round(total_sample / 2) + 1, np.round(total_sample / 2) + 1)
    t = np.arange(0, N)
    Hm = np.zeros(N)

    """
    Transfer func, it based on the fft domain
        It define based origin 0 [-(N/2)+1, N/2]. So it need to move [0, N] and the form is like as reflection 0(bias) [0, N/2-1], N/2(Nyquist Frequency), [N/2+1, N-1]
        In below example, N=32, 0(biase) [0, 15], 16(Nyquist Frequency), [17, 31]

        Method 1 Square
                ------------------------------------------------
                |             N/2                              |
                |         1    ---------                       |
                | h(k) = --- * >         H(m)e^(j*2*pi*m*k / N)|
                |         N    ---------                       |
                |             m=-(N/2)+1                       |
                | j(n) = complex(cos(n), sin(n))               |
                ------------------------------------------------
                * The sampling is started at "0", So shift from [-(N/2)+1, N/2] to [0, N]

        Method 2 Sine

                   1   sin(pi*k*K/N)
            h(k) = - * -------------
                   N   sin(pi*k*1/N)
    """

    """ Transfer func - Method 1 """
    n1 = np.arange(-N / 2 + 1, N / 2 + 1)  # [-N/2+1, N/2]

    # Hm is initial filter for size (NO value)
    Hm1 = Hm.copy()

    # find the origin(x=0)
    index_origin = np.where(n1 == 0)[0][0]

    # Hm1's origin is N/2-1, (0, N)
    Hm1[index_origin : index_origin + int(K / 2) + 1] = 1
    Hm1[index_origin - int(K / 2) : index_origin] = 1

    """
        Shift to Half of Total length
            - method1. hstack
                Hm1 = np.hstack([Hm1[index_origin:], Hm1[0:index_origin]])
            - method2. roll
                Hm1 = np.roll(Hm1, int(N/2)+1)
    """
    # method 2
    Hm1 = np.roll(Hm1, int(N / 2) + 1)
    """
        Figure 5-18 (a)
        nfft = 32
        K = 7 
    """
    hk = dftinv(Hm1)

    """ Transfer func - Method 2 """
    denominate = np.sin(np.pi * t / N)
    nominate = np.sin((bandwidth * 2 + 1) * np.pi * t / N)
    Hsin = np.zeros(len(t))
    for k, term in enumerate(nominate):
        if nominate[k] == 0 and denominate[k] == 0:
            Hsin[k] = (bandwidth * 2 + 1) / N
        else:
            Hsin[k] = nominate[k] / denominate[k] / N

    hsin = dftinv(Hsin)

    # Comparision between Method1. Square and Method2. Sine
    figure = plt.figure(figsize=(10, 10))
    ax = [0] * 2

    ax[0] = figure.add_subplot(211)
    ax[1] = figure.add_subplot(212)

    ax[0].plot(hk, "b.")
    ax[0].plot(hsin, "r*")
    ax[0].grid(True)
    ax[0].set_title("Time domain")

    ax[1].plot(Hm1, "b.")
    ax[1].plot(Hsin, "r*")
    ax[1].grid(True)
    ax[1].set_title("Frequency response")

    plt.show()
