"""Parallel Graphical Equalizer

    This is for parallel structure of filters.
    The details of test is on test.py.

    The conversion function time to compute under the python is below,
        
        <function escape_zero_difference at 0x7fa26a39b550> 0.111 ms
        <function WaveProcessor.process_time_domain at 0x7fa26a39b1f0> 8.209 ms

        32 band:
        <function genTargetResponse at 0x7fa26a39b700> 27.434 ms
        <function genPolePosition at 0x7fa26a39b820> 0.125 ms
        <function genZeroPosition at 0x7fa26a39b940> 9.757 ms
        
        64 band:
        <function genTargetResponse at 0x7fdd60e5b5e0> 41.191 ms
        <function genPolePosition at 0x7fdd60e5b700> 0.126 ms
        <function genZeroPosition at 0x7fdd60e5b820> 32.792 ms
    ---
    TODO LIST
    [ ] genPolePosition:
        [ ] 0. pole contains plus and minus, then which value?
        [ ] 1. interpolation method research and arrangement
                - linear, cubic, spline, etc.
                - sinc interpolation
                - reference. https://jaejunyoo.blogspot.com/2019/05/signal-processing-for-communications-9-1.html

    [ ] genZeroPostition:
        [ ] 0. Matrix M is not include imaginary, the paper said it is.
        [ ] 0. Why the third b coffcient is not exist ?
        [ ] 1. How to check the denominator is not correct? 
            -> Make the example
        [ ] 2. How to check the M is not correct?           
            -> Make the example
        [ ] 3. z = e^(-jwn) ?
            - Do Raw code for scratch FFT function
        [ ] 4. Normal LS design ?

    [ ] arrange the concept:
        - Why the windowing is applying two times which are in the time domain and after the ifft?
        - The difference between linear phase and minimum phase filer design
            - What is the Moore-Penrose pseudo-inverse, which is the concept for least-square solution?
        - [TODO]reason of pole position ~~ genPolePosition   
            - Fixed pole design         
                - Reference. http://home.mit.bme.hu/~bank/parfilt/

    [ ] merge:
        - graphical equalizer to filter_analayze

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.interpolate import pchip_interpolate

if __name__.split(".")[0] == "lib":
    from lib.config import *
    from lib.util import plot_audacity_freq_response
else:
    from config import *
    from util import plot_audacity_freq_response

if DEBUG:
    if __name__.split(".")[0] == "lib":
        from lib.debug.log import PRINTER
        from lib.debug.util import check_time, print_func_time
    else:
        from debug.log import PRINTER
        from debug.util import check_time, print_func_time


@check_time
def escape_zero_difference(array: np.ndarray):
    """
    """
    result_array = array.copy()

    zero_diff_location = np.where(np.diff(result_array) == 0)[0]
    while len(zero_diff_location) > 0:
        loc = zero_diff_location[0]
        if loc == len(result_array) - 1:
            pass
        if loc == 0:
            if result_array[loc] == result_array[loc + 1]:
                id_start = loc
                id_end = np.where(result_array[loc] < result_array)[0][0]
                result_array[id_start:id_end] = np.linspace(
                    result_array[id_start],
                    result_array[id_end],
                    len(result_array[id_start:id_end]),
                    endpoint=False,
                )
                for loc_local in range(id_start, id_end):
                    if loc_local in zero_diff_location:
                        zero_diff_location = np.delete(
                            zero_diff_location,
                            np.where(zero_diff_location == loc_local),
                        )
        elif result_array[loc + 1] == result_array[loc] == result_array[loc - 1]:
            id_start = loc - 1
            id_end = np.where(result_array[loc] < result_array)[0][0]
            result_array[id_start:id_end] = np.linspace(
                result_array[id_start],
                result_array[id_end],
                len(result_array[id_start:id_end]),
                endpoint=False,
            )
            for loc_local in range(id_start, id_end):
                if loc_local in zero_diff_location:
                    zero_diff_location = np.delete(
                        zero_diff_location, np.where(zero_diff_location == loc_local)
                    )
        else:
            result_array[loc] = (result_array[loc - 1] + result_array[loc + 1]) / 2
            zero_diff_location = zero_diff_location[1:]

    return result_array


@check_time
def genTargetResponse(
    origin: int,
    sampling_freq: int,
    cutoff_freq: np.array,
    gains: np.array,
    steps: float,
):
    """Target frequency response
        1. Find amplitudes of the target frequency using spline interpolation
            a. find the index of the target frqeuency in linear scale
                - set the origin for avoiding x to 0 in log10
            b. set gain in the target cut-off frequency
            c. Scale from linear to Logarithmic
            d. Cubic Hermite interpolation or spline interpolation in Logarithmic scale
                - [V] pchip
                - spline
        2. Find phase of the target frequency using Hilbert transform 
            a. resample in linear frequency
                - index the logarithmic frequency to linear frequency, id_log2linear
            b. avoid overlapped frequency bin in interpolation
            c. linear interpolation sampling in linear frequency
            d. compute logarithm
            e. minimum phase filter using Hilbert transform
                - include the reflection 
                - hilbert function in scipy = analytic signal -> hilbert transform = imag(hilbert function)
                - https://dsp.stackexchange.com/questions/42917/hilbert-transformer-and-minimum-phase
                - https://ccrma.stanford.edu/~jos/sasp/sasp.html
            f. Resample in Logarithmic frequency
                - compute reflection
        
        * Linear resolution = steps
        * Logarithmic resolution = the number of cuf-off frequency * 10
    """

    """ 1. Find amplitudes of the target frequency using spline interpolation """
    # 1-a. find the index of the target frqeuency in linear scale
    fs, fc, target_gain, num_step = sampling_freq, cutoff_freq, gains, steps
    f_linear = np.linspace(origin, fs // 2, num_step, endpoint=True)

    if DEBUG_PRINT:
        print("-" * 40)
        print("Target frequency: \n", fc)
        print("Frequency Resolution: ", (fs // 2) / num_step)

    # 1-b. set gain in the target cut-off frequency
    gain_fc = np.zeros(len(target_gain) + 2)
    gain_fc[0] = target_gain[0]
    gain_fc[1:-1] = target_gain
    gain_fc[-1] = target_gain[-1]  # Nyquist frequency

    # 1-c. Scale from linear to Logarithmic
    f_target_log = np.zeros(len(fc) + 2)
    f_target_log[0] = 0
    f_target_log[1:-1] = np.log10(fc)
    f_target_log[-1] = np.log10(f_linear[-1])
    f_target_log = escape_zero_difference(f_target_log)

    # 1-d. Cubic Hermite and spline interpolation in logarithmic scale
    f_target_intep_log = np.array([])
    gain_target_log = np.array([])
    for fid in range(len(f_target_log) - 1):
        buf_x = np.linspace(f_target_log[fid], f_target_log[fid + 1], num=11)
        buf_y = pchip_interpolate(f_target_log, gain_fc, buf_x)
        if len(f_target_intep_log) == 0:
            f_target_intep_log = np.append(f_target_intep_log, buf_x)
            gain_target_log = np.append(gain_target_log, buf_y)
        else:
            f_target_intep_log = np.append(f_target_intep_log, buf_x[1:])
            gain_target_log = np.append(gain_target_log, buf_y[1:])
    del buf_x, buf_y

    f_target_intep_log = 10 ** f_target_intep_log

    if DEBUG_PRINT:
        # The number of points = N(P-1)-(P-2) ~ N=10-11 -> 10P
        w_linear = 2 * np.pi * f_linear / fs
        print("-" * 40)
        print(
            f"Resolution in frequency domain, 10P {len(cutoff_freq)*10}: result {len(f_target_intep_log)}"
        )
        print(f"Resolution in linear domain: 2^{int(np.log2(len(w_linear)))}")

    if DEBUG_PLOT:
        plt.plot(
            f_target_intep_log, gain_target_log, label="target_interpolation_log_scale"
        )
        plt.xscale("log")
        plt.show()

    """ 2. Find phase of the target frequency using Hilbert transform """
    # 2-a. resample in linear frequency
    id_log2linear = (
        (f_target_intep_log - f_linear[0]) // abs(f_linear[1] - f_linear[0])
    ).astype(int)

    assert any(np.diff(id_log2linear) == 0)

    if not all(id_log2linear < num_step):
        raise Exception("Cutoff Frequency out of range")

    for i, id in enumerate(id_log2linear):
        if id >= len(f_linear) - 1:
            if abs((f_linear[int(id)] - f_linear[int(id_log2linear[i])]) > 1e-6):
                raise ValueError(
                    "index out of range", id_log2linear[i], f_linear[int(id)]
                )
        elif not (
            f_linear[int(id)]
            <= f_linear[int(id_log2linear[i])]
            <= f_linear[int(id) + 1]
        ):
            raise ValueError("index out of range", id_log2linear[i], f_linear[int(id)])
        else:
            pass

    f_target_linear = np.zeros_like(f_target_intep_log)
    f_target_linear = f_linear[id_log2linear]

    assert (
        f_target_linear[0] - f_linear[0] < 1e-6
        and f_target_linear[-1] - f_linear[-1] < 1e-6
    )
    del f_target_intep_log

    # 2-b. avoid overlapped frequency in interpolation, interpolation needs a increment
    f_target_linear = escape_zero_difference(f_target_linear)
    if DEBUG_PRINT:
        print("-" * 40)
        print(
            "Average Log- > Linear Error:  ",
            f_target_linear - [f_linear[int(id)] for id in id_log2linear],
        )

    # 2-c. Linear interpolation sampling in linear frequency
    gain_target_linear = np.interp(f_linear, f_target_linear, gain_target_log)

    # 2-d. compute logarithm and apply reflection,
    amp_target_linear = np.zeros(
        shape=len(gain_target_linear) * 2 - 2, dtype=np.complex128
    )
    amp_target_linear[: len(gain_target_linear)] = 10 ** (gain_target_linear / 20)
    amp_target_linear[len(gain_target_linear) :] = amp_target_linear[
        1 : len(gain_target_linear) - 1
    ][
        ::-1
    ]  # reflection. np.flip(amp_target_linear)

    # 2-e. minimum phase filter using hilbert transform, FFT and iFFT
    hilbert_transformer = hilbert(np.log(amp_target_linear).real).imag
    hilbert_transformer = np.exp(-1j * hilbert_transformer)

    if DEBUG_PLOT:
        amp_target_linear_plot = amp_target_linear.copy()

    amp_target_linear = amp_target_linear * hilbert_transformer
    amp_target_linear = amp_target_linear[: len(gain_target_linear)]

    if DEBUG_PLOT:
        _, axes = plt.subplots(nrows=2)
        axes[0].plot(20 * np.log10(np.abs(amp_target_linear_plot)), label="before")
        axes[0].plot(
            20 * np.log10(np.abs(amp_target_linear)), label="hilbert transform"
        )

        axes[1].plot(np.angle(amp_target_linear_plot, deg=True), label="before")
        axes[1].plot(np.angle(amp_target_linear, deg=True), label="hilbert transform")

        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlim(f_linear[0], f_linear[-1])
            ax.legend()
        plt.show()

    # 2-f. resample in logarithmic frequency
    size_target_freq = len(id_log2linear)
    target_amp = np.zeros(shape=size_target_freq * 2 - 2, dtype=np.complex128)
    target_freq = np.zeros(shape=size_target_freq * 2 - 2)
    target_amp[:size_target_freq] = amp_target_linear[id_log2linear]
    target_freq[:size_target_freq] = f_target_linear
    # reflection
    target_amp[size_target_freq:] = (target_amp[1 : size_target_freq - 1].conj())[::-1]
    target_freq[size_target_freq:] = (
        2 * f_linear[-1] - target_freq[1 : size_target_freq - 1]
    )[::-1]

    del id_log2linear, amp_target_linear

    if DEBUG_PLOT:
        _, axes = plt.subplots(nrows=2)
        fc_plot = np.zeros(len(fc) + 2)
        fc_plot[0] = origin
        fc_plot[1:-1] = fc
        fc_plot[-1] = int(sampling_freq / 2)

        axes[0].scatter(
            target_freq,
            20 * np.log10(np.abs(target_amp)),
            marker=".",
            label="interploated",
        )
        axes[0].scatter(fc_plot, gain_fc, color="r", marker="*", label="target gain")

        axes[1].scatter(
            target_freq,
            np.angle(target_amp, deg=True),
            label="interploated phase",
            marker=".",
        )
        for ax in axes:
            ax.set_xlim(f_linear[0], f_linear[-1])
            ax.set_xscale("log")
            ax.legend()
        plt.show()

    return target_freq, target_amp


@check_time
def genPolePosition(sampling_freq: int, cutoff_freq: np.array):
    """Generate pole position from cutoff frequency
        theta = 2*pi*fk/fs, for k= 1, 2, ..., N
        pole = e^{delta_theta_k/2}*e^{+-j*theta_k}
        * e^{delta_theta_k/2} means poles set at -3dB points
        delta_theta_1 = theta_2 - theta_1,
        delta_theta_k = (theta_k+1 - theta_k-1)/2,
        delta_theta_N = theta_N - theta_N-1

        ak,1 = (-p_k + p_k_conj) = -2|p_k|cos(theta_k)
        ak,2 = |p_k|^2
        * N is the number of cutoff frequency
    """
    theta = 2 * np.pi * cutoff_freq / sampling_freq
    delta_theta = np.zeros_like(theta)
    delta_theta[0] = theta[1] - theta[0]
    for id in range(1, len(theta) - 1):
        delta_theta[id] = (theta[id + 1] - theta[id - 1]) / 2
    delta_theta[-1] = theta[-1] - theta[-2]
    pole = np.exp(-1 * delta_theta / 2 - 1j * theta)
    a_k1 = -2 * np.abs(pole) * np.cos(theta)
    a_k2 = np.power(np.abs(pole), 2)

    assert len(a_k1) == len(a_k2) == len(cutoff_freq)
    return np.ones_like(a_k1), a_k1, a_k2


@check_time
def genZeroPosition(
    sampling_freq: int, a_coeff: tuple, target_freq: np.array, target_h: np.array
):
    """Generate zero position from target frequency and frequency response,
        High-Precision Parallel Graphical Eqaulizer
            z = e^(-jwn),
            H(z) = d0+ \sum_{k=1}^{N} ( b_{k,0} + b_{k,1} z^-1 ) / (1 + a_{k,1}*z^{-1} + a_{k,2}*z^{-2})
            h = Mp
            p = [b1,0, b1,1, ..., bK,0, bK,1, d0]^T
        M contains 1/(1 + ak,1 e^(-jwn) + ak,2 e^(-jwn)^2)) and e^(-jwn)/(1 + ak,1 e^(-jwn) + ak,2 e^(-jwn)^2))
        The last column of M belongs to the direct path gain d0, and thus all of its elements are one.
        The procedure is as below,
            1. denominator
                a. generate matrix z = e^(-jwn) for feedback coefficient 
                    array          : (1, e^(-jwn), e^(-jwn)^2)
                    shape          : (wn, a_coeff)
                b. compute feedback coefficient
                    array          : 1 / (1 + a_(k,1)*e^(-1jwn) + a_(k,2)*e^(-2jwn))
                    shape          : (wn, a_coeff) * (a_coeff, fc) = (wn, fc) 
            2. nominator and buffer
                a. compute feedforward coefficient
                    array          : [1/(1 + a_(k,1)*e^(-jwn) + a_(k,2)*e^(-jwn)^2)), e^(-jwn)/(1 + a_(k,1)*e^(-jwn) + a_(k,2)*e^(-jwn)^2)) , ... , 1]
                    shape          : (wn, fc) * (wn, 1, 2) = (wn, fc, 2)
                    expected result: (wn, fc*2+1) * (fc*2+1, 1) = (wn, 1) ~= hn
                b. change the sequence of coefficient and add buffer line as below,
                    array          : [b0,0, b0,1, b1,0, b1,1, b2,0, b2,1 ..., 1]
                    shape          : (wn, fc, 2) -> (wn, fc*2) -> append 1 -> (wn, fc*2+1) 
            3. computation for desired gain
                a. frequency weight applying target and cofficient matrix
                    weight         : 1/|ht|^(3/4)
                b. least-square solution
                    [V] Real impulse response   , p_opt = M^+ @ h_t, M^+ = (M^T @ M)^-1 @ M^T
                    [ ] Complex impulse response, p_opt = M^+ @ h_t  M^+ = (M^H @ M)^-1 @ M^H
    """
    fs = sampling_freq
    fn = target_freq[: len(target_freq) // 2 + 1]
    ht = target_h[: len(target_h) // 2 + 1]
    a, wn = np.array(a_coeff), 2 * np.pi * fn / fs

    """1. Denominator"""
    # 1-a. generate matrix z = e^(-jwn) for feedback coefficient
    denominator = np.array(  # (a_coeff, wn)
        [np.ones_like(wn), np.exp(-1j * wn), np.exp(-2j * wn)], dtype=np.complex128
    )
    denominator = denominator.transpose()  # (wn, a_coeff)

    # 1-b. compute feedback coefficient
    M_coeff = np.matmul(denominator, a)  # (wn, a_coeff) * (a_coeff, fc) = (wn, fc)
    M_coeff[np.where(M_coeff == 0)] = EPS
    M_coeff = 1 / M_coeff

    """2. Nominator and Buffer"""
    # 2-a. compute feedforward coefficient
    M_coeff = np.expand_dims(M_coeff, axis=-1)  # (wn, fc, 1)
    nominator = np.array(
        [np.ones_like(wn), np.exp(-1j * wn)], dtype=np.complex128
    )  # (wn, 2)
    nominator = np.expand_dims(nominator.transpose(), axis=1)  # (wn, 1, 2)
    M_coeff = np.matmul(M_coeff, nominator)  # (wn, fc, 1) * (wn, 1, 2) = (wn, fc, 2)

    # 2-b. change the sequence of coefficient and add buffer line
    M_coeff = M_coeff.reshape(
        (M_coeff.shape[0], M_coeff.shape[1] * M_coeff.shape[2])
    )  # (wn, fc*2  ) = [b0,0, b0,1, b1,0, b1,1, b2,0, b2,1 ...]
    M_coeff = np.append(  # (wn, fc*2+1) = [b0,0, b0,1, b1,0, b1,1, b2,0, b2,1 ..., 1]
        M_coeff, np.ones(shape=(M_coeff.shape[0], 1)), axis=-1
    )

    """3. Computation for desired gain"""
    # 3-a. frequency weight
    weight_wn = 1 / np.abs(ht)
    M_coeff_w = np.expand_dims(weight_wn, axis=-1) * M_coeff
    ht_w = weight_wn * ht

    # 3-b. least-square solution (the paper version)
    # p_opt = M+ @ h_t, M+ = (M^T @ M)^-1 @ M^T
    # M_r = np.array([M_coeff_w.real, M_coeff_w.imag])                     # (2, wn, fc*2+1), n = 289, fc = 31, dc
    # h_tr = np.array([ht_w.real, ht_w.imag])                              # (2, 289)
    # M_inv = np.matmul(M_r.transpose(0, 2, 1), M_r)                       # (2, 63, 289) @ (2, 289, 63) = (2, 63, 63)
    # M_inv[np.where(M_inv == 0)] = EPS
    # M_inv = np.linalg.inv(M_inv)                                         # (2, 63, 63), if complex, using np.linalg.pinv
    # M_r_plus = np.matmul(M_inv, M_r.transpose(0, 2, 1))                  # (2, 63, 63) @ (2, 63, 289) = (2, 63, 289)
    # p_opt = np.matmul(M_r_plus, np.expand_dims(h_tr, axis=-1))           # (2, 63, 289) @ (2, 289, 1) -> (2, 63, 1)
    # p_opt = p_opt[0]                                                     # extract real part
    # p_opt = p_opt.squeeze()

    # 3-b. least-square solution (compressed version)
    M_r = M_coeff_w.real  # (289, 63)
    h_tr = ht_w.real  # (289, )
    M_inv = np.matmul(M_r.T, M_r)  # (63, 289) @ (289, 63) = (63, 63)
    M_inv[np.where(M_inv == 0)] = EPS
    M_inv = np.linalg.inv(M_inv)  # (63, 63)
    M_r_plus = np.matmul(M_inv, M_r.T)  # (63, 63) @ (63, 289) = (2, 63, 289)
    p_opt = np.matmul(M_r_plus, h_tr)  # (63, 289) @ (289, 1) = (63, 1)

    if DEBUG_PLOT:
        Hn = np.matmul(
            M_coeff, np.expand_dims(p_opt, axis=-1)
        ).squeeze()  # (289, 63) @ (63, 1) -> (289, 1)
        error = np.abs(np.power(np.abs(Hn - ht), 2))
        print("error: ", np.max(error))

        _, axes = plt.subplots(nrows=3)
        amplitude_Hn = 20 * np.log10(np.abs(Hn))
        amplitude_ht = 20 * np.log10(np.abs(ht))
        angle_Hn = np.angle(Hn, deg=True)
        angle_ht = np.angle(ht, deg=True)

        axes[0].scatter(fn, amplitude_Hn, marker=".", label="Calculated Gain")
        axes[0].scatter(fn, amplitude_ht, color="r", marker="*", label="Target gain")
        axes[1].plot(fn, angle_Hn, label="Calculated phase")
        axes[1].plot(fn, angle_ht, color="r", label="Target phase")

        axes[2].scatter(
            fn, error, marker=".", label="Difference Gain",
        )

        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlim(fn[0], fn[-1])

        plt.show()
    del M_coeff
    # Extract b_coff
    d, p_opt = p_opt[-1], p_opt[:-1]
    b0, b1 = p_opt[0::2], p_opt[1::2]

    assert b0[0] == p_opt[0] and b0.shape == b1.shape
    return d, (b0, b1, np.zeros_like(b0))


class GraphicalEqualizer(object):
    """Graphical Equalizer
        H(z) = d + sum_k=1^K (bk,0 + bk,1z-1) / (1 + ak,0z-1 + ak,1z-2)
    """

    def __init__(
        self, sampling_freq, cufoff_freq: np.array, gain: np.array, Q: np.array
    ) -> None:
        self.sampling_freq = sampling_freq
        self.cufoff_freq = cufoff_freq
        self.gain = gain
        self.steps = 2 ** 15
        self.origin = 1
        self.axes = []
        self.freq_target, self.H_target = genTargetResponse(
            self.origin, self.sampling_freq, self.cufoff_freq, self.gain, self.steps
        )
        self.a = genPolePosition(self.sampling_freq, self.cufoff_freq)
        self.d, self.b = genZeroPosition(
            self.sampling_freq, self.a, self.freq_target, self.H_target
        )

    def write_to_file(self, save_path):
        with open(save_path, "w") as f:
            for ch in range(len(self.b[0])):
                # f.write(f'{ch}: ')
                f.write(f"{self.b[0][ch]} ")
                f.write(f"{self.b[1][ch]} ")
                f.write(f"{self.b[2][ch]} ")
                f.write(f"{self.a[0][ch]} ")
                f.write(f"{self.a[1][ch]} ")
                f.write(f"{self.a[2][ch]}\n")
            f.write(f"{self.d}\n")
        print(f"Written to file, {save_path}")


    def freqz(self, save_path="", full=False):
        """Apply parallel filter to data
                         -jw                  -jw              -jwM
                -jw   Bk(e  )    bk[0] + bk[1]e    + ... + bk[M]e
            H_k(e  ) = ------ = -----------------------------------
                         -jw                  -jw              -jwN
                    Ak(e  )    ak[0] + ak[1]e    + ... + ak[N]e
            
               -jwn                        -jwn           
            H(e  ) =  d +  sigma_k H_k(e  )
        """
        fs = self.sampling_freq
        fc_target, gain_target = self.freq_target, 20 * np.log10(np.abs(self.H_target))
        num_step = self.steps

        w = np.linspace(0, np.pi, num_step, endpoint=True)
        zm = np.array([np.ones_like(w), np.exp(-1j * w), np.exp(-1j * 2 * w)])
        a_coeff = np.array(self.a).T
        b_coeff = np.array(self.b).T
        d = self.d

        coeff = np.stack([b_coeff, a_coeff], axis=-1).transpose(0, 2, 1)
        h = coeff @ zm
        h = h.transpose(2, 0, 1)
        h = h[:, :, 0] / h[:, :, 1]
        h = np.sum(h, axis=-1)
        d = np.ones_like(h) * d
        h += d

        plt.scatter(
            w / (2 * np.pi) * fs,
            20 * np.log10(np.abs(h)),
            marker="*",
            color="k",
            label="simulation",
        )

        plt.scatter(
            fc_target[: len(fc_target) // 2 + 1],
            gain_target[: len(fc_target) // 2 + 1],
            marker="o",
            color="r",
            label="target",
        )

        if full:
            w_inv = np.linspace(np.pi, 2 * np.pi, num_step, endpoint=True)
            zm = np.array(
                [np.ones_like(w_inv), np.exp(-1j * w_inv), np.exp(-1j * 2 * w_inv)]
            )
            h_reflection = coeff @ zm
            h_reflection = h_reflection.transpose(2, 0, 1)
            h_reflection = h_reflection[:, :, 0] / h_reflection[:, :, 1]
            h_reflection = np.sum(h_reflection, axis=-1)
            d = np.ones_like(h_reflection) * d
            h_reflection += d

            plt.scatter(
                w / (2 * np.pi) * fs,
                20 * np.log10(np.abs(np.flip(h_reflection))),
                marker=".",
                color="b",
                label="simulation_inverse",
            )

        plt.xlim(1,)
        plt.xscale("log")
        plt.grid()
        plt.legend()

        if len(save_path)==0:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.show()


    def freqz_comp_audacity(self, save_path=""):
        x, y = plot_audacity_freq_response()
        plt.scatter(x, y, marker='.', color='dimgrey', label='after processing')

        plt.scatter(
            self.cufoff_freq,
            self.gain,
            marker="o",
            color="orangered",
            label="target"
        )

        _fontsize = 'large'
        plt.xlim(20,)
        plt.xscale("log")
        plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                ["20", "50", "100", "200", "500", "1K", "2K", "5K", "10K", "20K"],
                fontsize=_fontsize)
        plt.yticks(np.arange(-20, 40, 10) if min(self.gain) < 0 else np.arange(0, 40, 10),
                fontsize=_fontsize)
        plt.grid()
        plt.legend(fontsize="large")
        plt.xlabel("Frequency (Hz)",
                fontsize=_fontsize)
        plt.ylabel("Gain (dB)",
                fontsize=_fontsize)

        if len(save_path)==0:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.show()


    def plot_debug(self):
        if len(self.axes) == 0:
            print("No axes to plot")


if "__main__" == __name__:
    import os

    root = os.getcwd()
    # sampling frequnecy
    fs = 44100

    # cuf-off freuqency case 1
    fc = np.array(
        (
            20,
            25,
            31.5,
            40,
            50,
            63,
            80,
            100,
            125,
            160,
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
            4000,
            5000,
            6300,
            8000,
            10000,
            12500,
            16000,
            20000,
        )
    )

    # cutoff frequency case 2
    fB = np.array(
        [
            2.3,
            2.9,
            3.6,
            4.6,
            5.8,
            7.3,
            9.3,
            11.6,
            14.5,
            18.5,
            23.0,
            28.9,
            36.5,
            46.3,
            57.9,
            72.9,
            92.6,
            116,
            145,
            185,
            232,
            290,
            365,
            463,
            579,
            730,
            926,
            1158,
            1447,
            1853,
            2316,
        ]
    )
    fU = np.array(
        [
            22.4,
            28.2,
            35.5,
            44.7,
            56.2,
            70.8,
            89.1,
            112,
            141,
            178,
            224,
            282,
            355,
            447,
            562,
            708,
            891,
            1120,
            1410,
            1780,
            2240,
            2820,
            3550,
            4470,
            5620,
            7080,
            8910,
            11200,
            14100,
            17800,
            22050,
        ]
    )
    fL = np.zeros_like(fU)
    fL[0] = 17.5
    fL[1:] = fU[:-1]

    fc_twice = np.zeros((2, len(fc)))
    fc_twice[0, :] = np.append(10, fU[:-1])
    fc_twice[1, :] = fc
    fc_twice = fc_twice.reshape((fc_twice.shape[0] * fc_twice.shape[1],), order="F")

    # gain case 1
    gain_c1 = np.array(
        [
            12,
            12,
            10,
            8,
            4,
            1,
            0.5,
            0,
            0,
            6,
            6,
            12,
            6,
            6,
            -12,
            12,
            -12,
            -12,
            -12,
            -12,
            0,
            0,
            0,
            0,
            -3,
            -6,
            -9,
            -12,
            0,
            0,
            0,
        ]
    )

    # gain case 2
    gain_c2 = np.zeros_like(fc)
    gain_c2[0::2] = 12
    gain_c2[1::2] = -12

    # gain case 3
    gain_c3 = np.zeros_like(gain_c2)
    gain_c3[np.where(fc == 2000)] = 12

    # gain case 4
    gain_c4 = np.ones_like(gain_c2) * 12

    # gain case 5
    gain_c5 = np.zeros_like(fc)
    gain_c5[0::3] = 0
    gain_c5[1::3] = 0
    gain_c5[2::3] = 12

    # set the gain
    gains = [gain_c1, gain_c2, gain_c3, gain_c4, gain_c5]
    # for id_gain, gain in enumerate(gains):
    gain = gain_c5

    gain_twice = np.zeros((2, len(fc)))
    gain_twice[0, :] = gain
    gain_twice[1, :] = gain
    gain_twice = gain_twice.reshape(
        (gain_twice.shape[0] * gain_twice.shape[1],), order="F"
    )
    gain_twice[1:] = gain_twice[:-1]
    gain_twice[0] = 0

    # set the Q, not implemented yet
    Q = np.ones_like(fc)

    eq = GraphicalEqualizer(fs, fc, gain, Q)
    eq.freqz()
    # eq.freqz_comp_audacity()

    # file = "/lib/data/test_graphic_equalizer.txt"
    # eq.write_to_file(f"{root}/{file}")

    # print_func_time()
