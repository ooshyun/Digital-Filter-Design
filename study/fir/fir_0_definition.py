"""
    Book "Understanding Digital Signal Processing. Ch 5. 175 page
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import scipy.signal


def fir(frame, nTap):
    output = [0] * len(frame)
    if isinstance(frame, np.ndarray):
        for index, value in enumerate(frame):
            if index < (nTap - 1):
                output[index] = np.sum(frame[:index + 1]) / nTap
            else:
                output[index] = np.sum(frame[index - nTap + 1:index + 1]) / nTap
    else:
        pass

    return output


if __name__ == "__main__":
    """
        Finite Impulse Response Filters 
        fs: frequency response
        x: samples
        y: sin(2*pi*x / sampling frequency)
        _h: coefficient array
        
        Calculation of phase difference
            - In this case, being possible because it's single tone
            - Taking the maxium value, ane calculate below equation
                difference of samples between after and before filter / sampling frequecny * 360 degrees
                -> difference / total one frame * 360 degrees
        
        Get the Transfer Function Response
            - Transfer Function Response -> Applying to White noise has same effect
            - w, h = scipy.signal.freqz(b, a, worN=64, whole=True)
            - w: 0 ~ pi
            - h: frequecny response complex value
            - whole: 0 ~ pi normalization -> 0 ~ 2pi
 
    """
    fs = 32
    total_sample = 64
    x = np.arange(0, total_sample)
    y = np.sin(2 * np.pi * x / fs)
    T = 2 * np.pi / fs

    """ 
        FIR
        Same effect between fir function and np.convolve 
    """
    y_fir = fir(y, 5)
    _h = np.zeros(fs)
    _h[:5] = np.ones(5) * 1 / 5
    y_fir_convolve = np.convolve(_h, y)[:len(y)]

    """ Get the Phase difference"""
    index_max = np.array(np.where(y == np.max(y)))
    index_max_fir = np.array(np.where(y_fir == np.max(y_fir)))
    phase_diff = (index_max - index_max_fir) / fs * 360

    y_dft_scipy = fft(y_fir, n=64)
    y_dft = np.fft.fft(y_fir, n=64)

    """ Transfer function Response """
    w, h = scipy.signal.freqz(_h, 1, worN=64, whole=True)

    amplitude = np.abs(h)
    # 20 * np.log10(np.abs(h))  # Convert to dB

    """
        Same equation,
        angle = 180 * np.angle(h) / np.pi  # Convert to degrees
    """
    angle = np.angle(h, deg=True)

    fig = plt.figure(constrained_layout=True, figsize=(24, 7))
    subfigs = fig.subfigures(2, 1)

    axTop = subfigs[0].subplots(ncols=1, nrows=1)
    subfigs[0].set_facecolor('0.75')
    axTop.plot(x[:32], y[:32], "*")
    axTop.plot(x[:32], y_fir[:32], ".")
    # axTop.plot(y_fir_convolve[:32], "o")
    # axTop.plot(_h, "o")
    axTop.set_xlabel('sample')
    axTop.set_ylabel('amplitude')
    axTop.grid()
    subfigs[0].suptitle('Time domain', fontsize='x-large')

    axesBottom = subfigs[1].subplots(ncols=1, nrows=2)
    subfigs[1].set_facecolor('0.75')
    axesBottom[0].plot(x, amplitude, "*")
    axesBottom[0].grid()

    axesBottom[1].plot(x, angle, "*")
    axesBottom[1].grid()

    for index, ax in enumerate(axesBottom):
        ax.set_xlabel('sample')
        if index == 0:
            ax.set_ylabel('amplitude')
        else:
            ax.set_ylabel('phase')
    subfigs[1].suptitle('Frequency domain', fontsize='x-large')

    fig.suptitle("DFT Result", fontsize='xx-large')
    plt.show()

    """
    Principle of convolution
    1. Invert the sequence of the second array
    2. Move one by one and sigma of multiplication
    """
    # print(np.convolve([1, 2, 3], [0, 1, 0.5]))
    """
                1   2   3
      0.5   1   0
                0           = 0

            1   2   3
      0.5   1   0
            1   0           = 1
            
        1   2   3
        0.5 1   0
        0.5 2   0           = 2.5
                    
        1   2   3
            0.5 1   0
            1   3           = 4         
        
        1   2   3
                0.5   1   0
                1.5         = 1.5
                
                result = [0 1 2.5 4 1.5]
    """
