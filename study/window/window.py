import numpy as np
import matplotlib.pyplot as plt


def asinc(M, w):
    index_denominator_zero = np.where(np.sin(w / 2) == 0)
    result = np.zeros(shape=(len(w),))
    for iw, _w in enumerate(w):
        if iw in index_denominator_zero:
            result[iw] = M
        else:
            result[iw] = M * np.sin(M * _w / 2) / (M * np.sin(_w / 2))
    return result


def rectangle_window():
    M = 21
    nsample = np.arange(-M + 1, M)
    amplitude = np.zeros(shape=(len(nsample),))
    amplitude[: len(amplitude) // 2 + 1] = 1
    amplitude = np.roll(amplitude, len(amplitude) // 4)

    figure = plt.figure(figsize=(10, 7))
    ax = figure.add_subplot(111)

    ax.scatter(nsample, amplitude)
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("amplitude")
    ax.set_title("Zero-Phase Rectangle Window, M=21")

    plt.grid(True)
    plt.show()


def fourier_transform_of_the_rectangular_window():
    M = 11

    nfft = 256
    w = np.arange(-nfft / 2, nfft / 2, 1) / 256 * 2 * np.pi

    window_asinc_linear_phase = asinc(M, w)
    window_asinc_linear_phase = (
        np.exp(-1j * (M - 1) / 2 * w) * window_asinc_linear_phase
    )  # linear phase

    window_asinc_zero_phase = asinc(M, w)
    theta = np.array([0 if sample > 0 else 1 for sample in window_asinc_zero_phase])
    theta[np.argwhere(np.abs(w) < 2 * np.pi / M)] = 0  # Main lobe
    window_asinc_zero_phase = np.exp(1j * 2 * np.pi * theta) * asinc(M, w)  # zero phase

    figure = plt.figure(figsize=(10, 7))

    ax = [0] * 3

    ax[0] = figure.add_subplot(121)
    ax[1] = figure.add_subplot(222)
    ax[2] = figure.add_subplot(224)

    ax[0].plot(w, window_asinc_zero_phase)
    ax[0].set_yticks(np.arange(-4, 14, 2))
    ax[0].set_xlabel("Normalized Frequency w (rad/sample)")
    ax[0].set_ylabel("amplitude")
    ax[0].set_title("DFT Rectangle Window, M=11")
    ax[0].grid(True)

    ax[1].plot(w, 20 * np.log10(np.abs(window_asinc_zero_phase / M)))
    ax[1].set_yticks(np.arange(-40, 5, 5))
    ax[1].set_ylim((-40, 0))
    ax[1].set_xlabel("Normalized Frequency w (rad/sample)")
    ax[1].set_ylabel("amplitude (dB)")
    ax[1].set_title("DFT Rectangle Window, M=11")
    ax[1].grid(True)

    ax[2].plot(w, np.angle(window_asinc_zero_phase))
    ax[2].set_xlabel("Normalized Frequency w (rad/sample)")
    ax[2].set_ylabel("phase")
    ax[2].set_title("DFT Rectangle Window, M=11")
    ax[2].grid(True)

    plt.show()


def roll_off_of_the_rectangular_window_fourier_transform():
    M = 20

    nfft = 256
    w = np.arange(-nfft / 2, nfft / 2, 1) / 256 * 2 * np.pi
    w = w[len(w) // 2 + 1 :]  # avoid log10(0) in loglog graph

    window_asinc = asinc(M, w)
    window_asinc = np.exp(-1j * (M - 1) / 2 * w) * window_asinc

    loglog = (-np.log10(w) - 1) * -6.0206 / (-np.log10(0.2) - 1)

    figure = plt.figure(figsize=(10, 7))

    ax = figure.add_subplot(111)

    ax.plot(w, 20 * np.log10(np.abs(window_asinc / M)))
    ax.plot(w, loglog, "r--")
    ax.set_yticks(np.arange(-40, 5, 5))
    ax.set_ylim((-40, 0))
    ax.set_xlabel("Normalized Frequency w (rad/sample)")
    ax.set_ylabel("amplitude (dB)")
    ax.set_title("DTFT of Rectangle Window, M=20")
    ax.grid(True)

    plt.show()


if __name__ == "__main__":
    rectangle_window()
    fourier_transform_of_the_rectangular_window()
    roll_off_of_the_rectangular_window_fourier_transform()
