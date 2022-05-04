"""
    Book "Understanding Digital Signal Processing. Ch 5. 181 page
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.core.fromnumeric import argmax
from scipy.fft import fft, ifft
import scipy.io.wavfile as wav
import scipy.signal
from scipy.signal import get_window
from scipy.signal.windows.windows import hann

"""
여기서 부터 진행중!!!!
흐름: 필터를 만들자! > 딜레이를 줘서 주파수 반응을 봐보니 끝에 갈무리가 남아! > LPF를 만들고 싶어... 
                > 주파수에서 짤라서 시간도메인으로 역변환해서 시간 도메인에서 필터를 구해보자!
                > 필터가 정확하게 짤리지 않아 ㅠ.ㅠ... > 윈도우를 사용한다! > 윈도우와 transfer function을 주파수 도메인에서 비교해본다!
                > 시간도메인에서 평행이동은 주기함수라는 가정하에 주파수 도메인에 변화를 주지 않는다, 위상만 변화할 뿐!
                ---
                그럼 어떤 윈도우와 transfer function 이 있을까? 
                모든 필터는 디지털에서는 아날로그 필터를 시간도메인 혹은 주파수 영역에서 사용한다.
                연산 시간에 따라 주파수 영역에서 필터처리를 할 수 있고 혹은 시간 영역에서 필터처리를 할 수 있다.
                ---
                adaptive filter가 measurement에 따라 필터 coefficient를 변화시켜주는 필터를 의미한다.
                ---
                
    frequency domain treatment 
    1. convolution: frequency domain filter ->ifft, idft -> time filter -> apply
    2. fft: signal, filter convert to frequeuncy domain using fft -> multiplication
    -> fft is usually used above 64 size frame 
    if making convolute == fft
    -> when you get the fft, zero padding should be because convolute get 2 times length
"""

from plotter import *

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
    return [
        sum((h[k] * iexp(-2 * math.pi * i * k / n) for k in range(n))) for i in range(n)
    ]


def dftinv(h):
    """
        naive idft(0 ~ N)
    """
    n = len(h)
    return [
        sum((h[k] * iexp(2 * math.pi * i * k / n) for k in range(n))) / n
        for i in range(n)
    ]


def blackmanWindow_naive(size: int):
    return np.array(
        [
            0.42
            - 0.5 * np.cos(2 * np.pi * k / (size - 1))
            + 0.08 * np.cos(4 * np.pi * k / (size - 1))
            for k in range(size)
        ]
    )


def blackmanWindow(size: int, sym: bool = False):
    return scipy.signal.windows.blackman(size, sym=sym)


def chebyshevWindow(size: int, sym: bool = False):
    return scipy.signal.windows.chebwin(size, at=45, sym=sym)


def kaiserWindow(size: int, beta: float = 4, sym: bool = False):
    return scipy.signal.windows.kaiser(size, beta, sym=sym)


class TransResponse(object):
    def __init__(self, name, transferFunc, size, logscale, symmetric, shift):
        self.name = name
        self.xaxis = None
        self.amplitudes = None
        self.phases = None

        self.transfunc = transferFunc
        self.size = size

        self.logscale = logscale
        self.symmetric = False
        self.shift = shift

        self.calculateTransferResponse()

    def calculateTransferResponse(self):
        self.xaxis, self.amplitudes, self.phases = transfer_response(
            self.transfunc,
            self.size,
            logscale=self.logscale,
            half=self.symmetric,
            shift=self.shift,
        )


class FreqResponse(object):
    def __init__(self, name, transferFunc, size, logScale, symmetric, shift):
        self.name = name
        self.xaxis = None
        self.amplitudes = None
        self.phases = None

        self.transfunc = transferFunc
        self.size = size

        self.logScale = logScale
        self.symmetric = symmetric
        self.shift = shift

        self.calculateTransferResponse()

    def calculateFreqResponse(self):
        self.xaxis, self.amplitudes, self.phases = transfer_response(
            self.transfunc,
            self.size,
            logscale=self.logScale,
            half=self.symmetric,
            shift=self.shift,
        )


if __name__ == "__main__":
    """
        For Frequency Response about filtered wave file
    """

    window_dict = {}
    rectangle = np.ones(N) / N

    # samplingFreq = N
    window_dict["Rectangle"] = rectangle
    window_dict["Blackman Window"] = rectangle * blackmanWindow(N, sym=False)
    window_dict["Blackman Window_naive"] = rectangle * blackmanWindow_naive(N)
    window_dict["Chebyshev Window"] = rectangle * chebyshevWindow(N, sym=False)
    window_dict["Kaiser Window"] = rectangle * kaiserWindow(N, sym=False)

    def plot_all(datas: list):
        for data in datas:
            print(data)
            plt.plot(data, ".")
        plt.grid()
        plt.show()

    # Check the alignment
    # sw(list(window_dict.values()))

    def zero_padding(h, n: int):
        assert len(h) % 2 == 0

        if len(h) >= n:
            return h
        else:
            transfer = h.copy()
            n_dft = n - len(h)
            zero_padding = np.zeros((n_dft) // 2)
            # bias = transfer[0]
            # transfer = transfer[1:]
            transfer = np.append(zero_padding, transfer)
            transfer = np.append(transfer, zero_padding)
            # transfer = np.append(bias, transfer)
            return transfer

    # for label, data in zip(window_dict.keys(), window_dict.values()):
    #     buf = zero_padding(data, n=512)
    #     buf = np.roll(buf, len(buf)//2)
    #     window_dict[label] = dft(buf)

    # plot_all(list(window_dict.values()))

    sr, white_noise = wav.read("./ExampleMusic/White Noise.wav")

    bit_depth = 0
    if white_noise.dtype == "int16":
        bit_depth = 15
    elif white_noise.dtype == "int32":
        bit_depth = 31
    elif white_noise.dtype == "float32":
        bit_depth = 0
    elif white_noise.dtype == "float64":
        bit_depth = 0

    white_noise = white_noise / (2 ** bit_depth)
    # white_noise = abs(white_noise)

    # test = np.sin(2 * np.pi * 1000 * np.arange(0, 10 * 44100) / 44100, dtype=np.float32) + np.sin(2 * np.pi * 5000 * np.arange(0, 10 * 44100) / 44100, dtype=np.float32)
    # white_noise = test

    frame_size = 1024
    num_chunk = len(white_noise) // frame_size
    fft_white_noise = np.zeros((num_chunk, frame_size), dtype=np.complex128)

    # hanning = np.array([(0.5 - (0.5 * math.cos((2 * math.pi * i) / (frame_size - 1)))) for i in range(frame_size)],dtype='float64')

    for idx in range(num_chunk):
        buf = fft(white_noise[idx * frame_size : (idx + 1) * frame_size], n=frame_size)
        fft_white_noise[idx] = buf

    freq = np.arange(2, frame_size // 2 + 2) * sr // frame_size
    y = fft_white_noise[:, : fft_white_noise.shape[-1] // 2 + 1]

    # Full fft
    # freq = np.arange(1, frame_size+1)*sr//frame_size
    # y = fft_white_noise

    y = np.abs(y)
    y = np.sum(y, axis=0) / y.shape[0]
    plt.plot(freq, y[1:])
    plt.show()

    transresponse = {}
    # name, transferFunc, size, logScale, symmetric, shift)
    # for label, window in zip(window_dict.keys(), window_dict.values()):
    #     transresponse[label] = TransResponse(name=label ,transferFunc=window, size=samplingFreq, logScale=False, symmetric=False, shift=True)

    plot_list = {
        "object": [],
        "amplitude": [],
        "phase": [],
        "amplitudeTime": [],
        "numbers": 0,
        "xlen": None,
        "plotStyle": "*",
    }

    # xaxis = transresponse_rectangle.xaxis

    # plot_list['xlen'] = int(samplingFreq/2)

    # plot_list['object'].append(transresponse_blackman)
    # # plot_list['object'].append(transresponse_kaiser)
    # # plot_list['object'].append(transresponse_chebyshev)
    # plot_list['object'].append(transresponse_rectangle)

    # plot_list['amplitudeTime'].append(w_blackman)
    # # plot_list['amplitudeTime'].append(w_kaiser)
    # # plot_list['amplitudeTime'].append(w_cheb)
    # plot_list['amplitudeTime'].append(w_rectangle)

    # plot_list['amplitude'].append(transresponse_blackman.amplitudes)
    # # plot_list['amplitude'].append(transresponse_kaiser.amplitudes)
    # # plot_list['amplitude'].append(transresponse_chebyshev.amplitudes)
    # plot_list['amplitude'].append(transresponse_rectangle.amplitudes)

    # # plot_list['amplitude'].append(amplitude_h2)

    # plot_list['phase'].append(transresponse_blackman.phases)
    # # plot_list['phase'].append(transresponse_kaiser.phases)
    # # plot_list['phase'].append(transresponse_chebyshev.phases)
    # plot_list['phase'].append(transresponse_rectangle.phases)

    # # plot_list['phase'].append(angle_h2)

    # plot_list['numbers'] = len(plot_list['phase'])

    # # plot_time_domain(t, **plot_list)
    # # plot_time_domain_compare(t, **plot_list)
    # # plot_tranfer_response(xaxis, **plot_list)
    # # plot_transfer_response_compare(xaxis, **plot_list)
