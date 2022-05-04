import math
import scipy
import matplotlib.pyplot as plt
from scipy.fftpack import *
import numpy as np
import matplotlib.pyplot as plt


def example_2ndinterpolation():
    """Example for interpolation
    """
    from scipy.interpolate import (
        PchipInterpolator,
        pchip_interpolate,
        CubicHermiteSpline,
        CubicSpline,
    )

    fig, ax = plt.subplots(figsize=(6.5, 5))

    def pchip_interpolation():
        # pchip_interpolate
        x_observed = np.linspace(0.0, 10.0, 11)
        y_observed = np.sin(x_observed)
        x = np.linspace(min(x_observed), max(x_observed), num=100)
        y = pchip_interpolate(x_observed, y_observed, x)
        ax.plot(x_observed, y_observed, "o", label="observation")
        ax.plot(x, y, label="pchip interpolation")

    def Hermite_spline_interpolation():
        # cubic Hermite and spline interpolation methods
        x = np.arange(10)
        y = np.sin(x)
        cs = CubicSpline(x, y)
        xs = np.arange(-0.5, 9.6, 0.1)
        ax.plot(x, y, "o", label="data")  # datapoint
        ax.plot(xs, np.sin(xs), label="true")  # origin
        ax.plot(xs, cs(xs), label="S")  # y
        ax.plot(xs, cs(xs, 1), label="S'")  # delta_y
        ax.plot(xs, cs(xs, 2), label="S''")  # delta_delta_y
        ax.plot(xs, cs(xs, 3), label="S'''")  # delta_delta_delta_y

    pchip_interpolation()
    # Hermite_spline_interpolation()
    plt.legend()
    plt.show()


def hilbert_from_scratch(u):
    # N : fft length, M : number of elements to zero out
    # U : DFT of u, v : IDFT of H(U)
    N = len(u)

    # take forward Fourier transform
    U = fft(u)
    M = N - N // 2 - 1

    # zero out negative frequency components
    U[N // 2 + 1 :] = [0] * M

    # double fft energy except @ DC0
    U[1 : N // 2] = 2 * U[1 : N // 2]

    # take inverse Fourier transform
    v = ifft(U)
    return v


def test_hilbert_from_scratch_time_domain():
    """Test hilbert transform from time-domain signal
    """
    N = 32
    f = 1
    dt = 1.0 / N
    y = []
    for n in range(N):
        x = 2 * math.pi * f * dt * n
        y.append(2 * math.sin(x))
    z1 = hilbert_from_scratch(y)
    z2 = hilbert(y)
    z3 = scipy.signal.hilbert(y)

    print(" n     y fromscratch scipy")
    for n in range(N):
        print(
            "{:2d} {:+5.2f} {:+10.2f} {:+5.2f} {:+5.2f}".format(
                n, y[n], z1[n], z2[n], z3[n]
            )
        )

    t = np.arange(0, N)

    y_fft = fft(y)
    y_fft = y_fft[: len(y_fft) // 2 + 1]

    z1_fft = fft(z1)
    z2_fft = fft(z2)
    z3_fft = fft(z3)

    z1_fft = z1_fft[: len(z1_fft) // 2 + 1]
    z2_fft = z2_fft[: len(z2_fft) // 2 + 1]
    z3_fft = z3_fft[: len(z3_fft) // 2 + 1]

    fig, ax = plt.subplots(nrows=3)
    ax[0].plot(t, y, label="y")
    # ax[0].plot(t, z1.imag, label='z1')
    ax[0].plot(t, z2, label="z2")
    # ax[0].plot(t, z3, label='z3')

    t = t[: len(t) // 2 + 1]
    ax[1].plot(t, 20 * np.log10(np.abs(y_fft)), label="y")
    # ax[1].plot(t, np.abs(z1_fft), label='z1')
    ax[1].plot(t, 20 * np.log10(np.abs(z2_fft)), label="z2")
    # ax[1].plot(t, np.abs(z3_fft), label='z3')

    ax[2].plot(t, np.angle(y_fft, deg=True), label="y")
    # ax[2].plot(t, np.angle(z1_fft, deg=True), label='z1')
    ax[2].plot(t, np.angle(z2_fft, deg=True), label="z2")
    # ax[2].plot(t, np.angle(z3_fft, deg=True), label='z3')

    plt.show()


def test_hilbert_from_scratch_frequency_response():
    """Test hilbert transform from frequency-domain amplitude
        Reference. Understanding Digital Signal Processing 441 page
    """
    N = 32
    f = np.arange(1, 5)
    dt = 1.0 / N
    y = []
    for n in range(N):
        y_one = sum([math.sin(2 * math.pi * freq * dt * n) / len(f) for freq in f])
        y.append(y_one)
    y_fft = np.fft.fft(y)
    data_fft = y_fft

    # Method 1. iFFT -> disgard imag -> hilbert -> FFT
    y_fft_real_HT = data_fft.copy() * 2.0
    y_fft_real_HT[len(y_fft_real_HT) // 2 + 1 :] = 0
    y_fft_real_HT[0] /= 2
    y_fft_real_HT[len(y_fft_real_HT) // 2] /= 2

    y_HT = np.fft.ifft(y_fft_real_HT)

    # y_HT = scipy.signal.hilbert(y_HT.real)
    y_fft_HT = np.fft.fft(y_HT)

    # Method 2. irFFT -> hilbert -> rFFT
    y_HT2 = np.fft.irfft(y_fft_real_HT[: len(data_fft // 2 + 1)], n=len(data_fft))
    # y_HT2 = scipy.signal.hilbert(y_HT2)
    y_fft_HT2 = np.fft.rfft(y_HT2)

    x = np.arange(N)
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
    ax0.plot(np.abs(data_fft), "*", label="fft signal")
    ax0.plot(np.abs(y_fft_HT), "x", label="fft signal_Hilbert Transform")
    ax0.plot(np.abs(y_fft_HT2), "s", label="rfft signal_Hilbert Transform")
    ax0.legend()

    ax1.plot(np.angle(data_fft, deg=True))
    ax1.plot(np.angle(y_fft_HT, deg=True))
    ax1.plot(np.angle(y_fft_HT2, deg=True))
    fig.tight_layout()

    ax2.plot(y, "*", label="fft signal")
    ax2.plot(y_HT.imag, "o", label="fft signal_Hilbert Transform")
    # ax2.plot(y_HT2, "s", label='rfft signal_Hilbert Transform')
    ax2.legend()
    plt.show()


def example_hilbert_signal():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import hilbert, chirp

    duration = 1.0
    fs = 400.0
    samples = int(fs * duration)
    t = np.arange(samples) / fs

    # We create a chirp of which the frequency increases from 20 Hz to 100 Hz and
    # apply an amplitude modulation.

    signal = chirp(t, 20.0, t[-1], 100.0)
    signal *= 1.0 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t)

    # The amplitude envelope is given by magnitude of the analytic signal. The
    # instantaneous frequency can be obtained by differentiating the
    # instantaneous phase in respect to time. The instantaneous phase corresponds
    # to the phase angle of the analytic signal.

    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs

    fig, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.plot(t, signal, label="signal")
    ax0.plot(t, amplitude_envelope, label="envelope")
    ax0.set_xlabel("time in seconds")
    ax0.legend()
    ax1.plot(t[1:], instantaneous_frequency)
    ax1.set_xlabel("time in seconds")
    ax1.set_ylim(0.0, 120.0)
    fig.tight_layout()
    plt.show()


def example_hilbert_transform():
    t = np.linspace(0, 6, 1000)
    f = 1000
    w0 = 2 * np.pi * f
    y = np.cos(w0 * t)
    axis = -1
    N = y.shape[axis]
    y_analytic = scipy.signal.hilbert(y)
    y_hilbert = y_analytic.imag

    fft_y = np.fft.fft(y)
    fft_hilbert = np.fft.fft(y_hilbert)
    fft_hilbert = np.fft.rfft(y_hilbert)
    fft_analytic = np.fft.fft(y_analytic)

    t = np.linspace(0, 6, 1000)
    f = 3000
    w0 = -2 * np.pi * f
    y = np.cos(w0 * t)
    y_effect = y - 1j * np.sin(w0 * t)

    fft_y = np.fft.fft(y)
    fft_effect = np.fft.fft(y_effect)

    fig, ax = plt.subplots(nrows=2, ncols=2)

    ax[0][0].plot(t, fft_y, label="amplitude")
    ax[1][0].plot(t, fft_effect, label="amplitude")

    ax[0][1].plot(t, np.angle(fft_y, deg=True), label="phase")
    ax[1][1].plot(t, np.angle(fft_effect, deg=True), label="phase")

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # ax.plot(t, np.ones_like(t)*2, y_analytic.imag)
    # ax.plot(t, y_analytic.real, y_analytic.imag)
    # ax.plot(t, y_analytic.real, np.ones_like(t)*(-2))

    # ax.set_xlim(-1, max(t)+1)
    # ax.set_ylim(-2, 2)
    # ax.set_zlim(-2, 2)

    # _, (ax0, ax1) = plt.add_subplots(nrows=2, ncols=2)

    # ax0[0].plot(t, y, "*", label='y')
    # ax0[0].plot(t, y_analytic, "*", label='analytic signal')

    # ax1[0].plot(t, y, label='y')
    # ax1[0].plot(t, y_analytic.imag, label='analytic signal')

    # ax0[1].plot(t, np.abs(fft_y), "*", label='y')
    # ax0[1].plot(t, np.abs(fft_analytic), "*", label='analytic signal')
    # ax0[1].plot(np.abs(fft_hilbert), "*", label='hilbert transform')

    # ax1[1].plot(np.angle(fft_y), label='y')
    # ax1[1].plot(t, np.angle(fft_analytic), label='analytic signal')
    # ax1[1].plot(np.angle(fft_hilbert), label='hilbert transform')

    # for _ax0, _ax1 in zip(ax0, ax1):
    #     for ax in (_ax0, _ax1):
    #         ax.legend()

    plt.show()


def test_even_odd_cross_matrix_1d():
    """Test (5, 3, 1) <-> (5, 2) in 1D
        (5, 3, 1) * (5, 1, 2) = (5, 3, 2) 
    """
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.power(x, 2)
    print(f"array {x}, {y}")

    z = np.append(x, y, axis=-1)
    print(f"append matrix {z}")

    z = z.reshape(len(z) // len(x), len(x)).transpose().flatten()
    print(f"cross matrix {z}")

    z_2 = np.array([x, y]).flatten("F")
    print(f"cross matrix 2 {z_2}")

    z_3 = np.array([x, y]).reshape((len(x) + len(y),), order="F")
    print(f"cross matrix 3 {z_3}")


def test_even_odd_cross_matrix_2d():
    """Test (5, 3, 1) <-> (5, 2) in 2D
        (5, 3, 1) * (5, 1, 2) = (5, 3, 2) 
    """
    print("\nBasic Test")
    x = np.array([np.arange(0, 16) for _ in range(32)])
    x = np.expand_dims(x, axis=-1)
    y = np.ones(shape=(32, 3))
    y[:, 0] = 0
    y[:, 2] = 2
    y = np.expand_dims(y, axis=1)
    z = np.matmul(x, y)
    print(f"{x.shape} @ {y.shape} = {z.shape}")
    # for i in range(16): print(f'{x[0, i, :]} * {y[i, :]} = {z[0, i, :]}')
    z = z.reshape((z.shape[0], z.shape[1] * z.shape[2]))
    print(z[0])

    print("\nTest graphic equalizer case")
    y = np.array([np.zeros((32,)), np.ones((32,)), np.ones((32,)) * 2]).transpose()
    y = np.expand_dims(y, axis=1)
    z = np.matmul(x, y)
    print(f"{x.shape} @ {y.shape} = {z.shape}")

    # for i in range(16): print(f'{x[0, i, :]} * {y[i, :]} = {z[0, i, :]}')

    z = z.reshape((z.shape[0], z.shape[1] * z.shape[2]))
    z = np.append(z, np.ones((z.shape[0], 1)) * 1000, axis=1)
    print(z[0])


if __name__ == "__main__":

    """ Non-linear, linear interpolation test """
    # example_2ndinterpolation()
    """ Hilbert Transform test """
    # test_hilbert_from_scratch_time_domain()
    # test_hilbert_from_scratch_frequency_response()
    # example_hilbert_signal()
    # example_hilbert_transform()
    """ Matrix Transform test"""
    # test_even_odd_cross_matrix_1d()
    # test_even_odd_cross_matrix_2d()

    pass
