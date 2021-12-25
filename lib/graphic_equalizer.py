"""Parallel Graphical Equalizer

    This is for parallel structure of filters.
    The details of testing is on test.py.
    ---
    TODO LIST
    [ ] genPolePosition:
        [ ] 0. pole contains plus and minus, then which value?

    [ ] genZeroPostition:
        [ ] 0. Matrix M is not include imaginary, the paper said it is.
        [ ] 0. Why the third b coffcient is not exist ?
        [ ] 1. How to check the denominator is not correct? 
            -> Make the example
        [ ] 2. How to check the M is not correct?           
            -> Make the example
        [ ] 3. z = e^(-jwn) ?
        [ ] 4. Normal LS design ?
        [ ] 5. While processing 61 channel, the 10Hz frequency missed
    [ ] write_to_file:
        [ ]- it did not worked
"""
# import math
# import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.interpolate import pchip_interpolate
# from scipy.fftpack import hilbert as hilbert_scipy

DEBUG_PRINT = False
DEBUG_PLOT = False


def plot_audacity_freq_response():
    root = os.getcwd()
    origin = os.path.join(root, 'lib/data/audacity_origin.txt')
    process = os.path.join(root, 'lib/data/audacity_filtered.txt')
    files = [origin, process]

    graphs = []
    for idx, file in enumerate(files):
        f = open(file, 'r').read().split('\n')[1:-1]
        graph = []
        for i in range(len(f)):
            freq, amp = f[i].split('\t')
            graph.append(np.array([float(freq), float(amp)]))
        graph = np.array(graph).T
        graphs.append(graph)
    return graphs[1][0], graphs[1][1] - graphs[0][1]

def genTargetResponse(origin: int, sampling_freq: int, cutoff_freq: np.array,\
                     gains: np.array, steps: float):
    """Target frequency response
        1. Find amplitudes of the target frequency using spline interpolation
            a. find the index of the target frqeuency in linear scale
            b. set gain in the target cut-off frequency
            c. Scale from linear to Logarithmic
            d. Cubic Hermite and spline interpolation in Logarithmic scale
                - [V] pchip
                - spline
        2. Find phase of the target frequency using Hilbert transform 
            a. resample in linear frequency
            b. avoid overlapped frequency bin in interpolation
            c. Cubic Hermite and spline interpolate in linear frequency
            d. compute logarithm
            e. Hilbert transform using FFT and iFFT
                - method 1 fft, truncate imaginary part
                - [V] method 2 rfft
            f. Resample in Logarithmic frequency
        * Linear resolution = steps
        * Logarithmic resolution = the number of cuf-off frequency * 10
        * In Hilber Transform, Didn't disgarding imaginary part occur error?
            - How to use or make the complex because after fft, the complex part doesn't go to hilbert.
                Hilbert transfrom allows real number.
            - KeyPoint in Hilbert
                - The negative frequency component is disrupted.
                - After FFT, iFFT, the amplitude is same and phase changes
    """
    """ 1. Find amplitudes of the target frequency using spline interpolation """
    # 1-a. find the index of the target frqeuency in linear scale
    fs, fc, target_gain, num_step = sampling_freq, cutoff_freq, gains, steps
    f_linear = np.linspace(origin, int(fs / 2),
                           num_step)    # min 10Hz, max fs/2, min >= 1 for log10
    w_linear = 2 * np.pi * f_linear / fs

    if DEBUG_PRINT:
        print('-' * 40)
        print('Target frequency: \n', fc)

    # 1-b. set gain in the target cut-off frequency
    gain_fc = np.zeros(len(target_gain) + 2)
    gain_fc[1:-1] = target_gain
    gain_fc[0] = target_gain[0]    # DC
    gain_fc[-1] = 0    # Nyquist

    index_fc = ((fc - f_linear[0]) // (f_linear[1] - f_linear[0])).astype(int)
    if not all(index_fc < num_step):
        raise Exception('Cutoff Frequency out of range')
    for i, id in enumerate(index_fc):
        if not (f_linear[id] <= fc[i] <= f_linear[id + 1]):
            raise ValueError(
                f'index out of range {f_linear[id]} <= {fc[i]} <= {f_linear[id+1]}'
            )
        else:
            pass

    # 1-c. Scale from linear to Logarithmic
    f_log = np.zeros_like(f_linear)
    f_log = np.log10(f_linear)    # avoid 0 in log10
    f_target_log = np.zeros(len(index_fc) + 2)
    for i, id in enumerate(index_fc):
        f_target_log[i + 1] = f_log[id]
    f_target_log[0] = f_log[0]
    f_target_log[-1] = f_log[-1]

    if DEBUG_PRINT:
        print('-' * 40)
        print(
            f'Average Linear -> Log Error: {np.average(abs(10**f_target_log[1:-1] - fc))} in target frequency'
        )

    # 1-d. Cubic Hermite and spline interpolation in Logarithmic scale
    f_target_intep_log = np.array([])
    gain_target_log = np.array([])
    for id in range(len(f_target_log) - 1):
        buf_x = np.linspace(f_target_log[id], f_target_log[id + 1], num=10)
        buf_y = pchip_interpolate(f_target_log, gain_fc, buf_x)
        if len(f_target_intep_log) == 0:
            f_target_intep_log = np.append(f_target_intep_log, buf_x)
            gain_target_log = np.append(gain_target_log, buf_y)
        else:
            f_target_intep_log = np.append(f_target_intep_log, buf_x[1:])
            gain_target_log = np.append(gain_target_log, buf_y[1:])
    del buf_x, buf_y
    f_target_intep = 10**f_target_intep_log

    assert abs(f_target_intep[-1] - f_linear[-1]) < 1e-6
    if DEBUG_PRINT:
        print('-' *
              40)    # The number of points = N(P-1)-(P-2) ~ N=10-11 -> 10P
        print(
            f'Resolution in frequency domain, 10P {len(cutoff_freq)*10}: result {len(f_target_intep)}'
        )
        print(f'Resolution in linear domain: 2^{int(np.log2(len(w_linear)))}')

    if DEBUG_PLOT:
        # plt.plot(f_target_intep, gain_target_log, label='target_interpolation_log_scale')
        # plt.xscale('log')
        pass
    """ 2. Find phase of the target frequency using Hilbert transform """

    # 2-a. resample in linear frequency
    index_log2linear = ((f_target_intep - f_linear[0]) // abs(
        (f_linear[1] - f_linear[0]))).astype(int)

    assert any(np.diff(index_log2linear) == 0)

    if not all(index_log2linear < num_step):
        raise Exception('Cutoff Frequency out of range')

    for i, id in enumerate(index_log2linear):
        if id >= len(f_linear) - 1:
            if abs((f_linear[int(id)] -
                    f_linear[int(index_log2linear[i])]) > 1e-6):
                raise ValueError('index out of range', index_log2linear[i],
                                 f_linear[int(id)])
        elif not (f_linear[int(id)] <= f_linear[int(index_log2linear[i])] <=
                  f_linear[int(id) + 1]):
            raise ValueError('index out of range', index_log2linear[i],
                             f_linear[int(id)])
        else:
            pass

    f_target_linear = np.zeros_like(f_target_intep)

    for i, id in enumerate(index_log2linear):
        f_target_linear[i] = f_linear[int(id)]

    # 2-b. avoid overlapped frequency bin in interpolation
    for i in np.argwhere(np.diff(f_target_linear) == 0):
        f_target_linear[i] = (f_target_linear[i - 1] +
                              f_target_linear[i + 1]) / 2

    if DEBUG_PRINT:
        print('-' * 40)
        print(
            "Average Log- > Linear Error:  ",
            np.average(
                abs(f_target_linear -
                    [f_linear[int(id)] for id in index_log2linear])))

    # 2-c. Cubic Hermite and spline interpolate in linear frequency
    gain_target_linear = np.array([])
    for id in range(len(f_target_linear) - 1):
        buf_x = np.linspace(f_target_linear[id],
                            f_target_linear[id + 1],
                            num=int(index_log2linear[id + 1] -
                                    index_log2linear[id] + 1))
        buf_y = pchip_interpolate(f_target_linear, gain_target_log, buf_x)
        if len(gain_target_linear) == 0:
            gain_target_linear = np.append(gain_target_linear, buf_y)
        else:
            gain_target_linear = np.append(gain_target_linear, buf_y[1:])
    del buf_x, buf_y

    # 2-d. compute logarithm
    amp_target_linear = 10**(gain_target_linear / 20)
    del gain_target_linear

    # 2-e. Hilbert transform using FFT and iFFT, method 2. rfft
    amp_target_linear_hilbert = np.fft.rfft(
        hilbert(
            np.fft.irfft(amp_target_linear,
                         n=(len(amp_target_linear) - 1) * 2)))
    if DEBUG_PLOT:
        # Method 1. fft, and comparision method 1 and 2
        buf = amp_target_linear.copy()
        buf_ht = np.zeros((len(amp_target_linear) - 1) * 2)
        buf_ht[:len(buf_ht) // 2 + 1] = buf
        buf_ht[0] /= 2
        buf_ht[len(buf_ht) // 2] /= 2
        buf_ifft = np.fft.ifft(buf_ht).real
        buf_ht_ifff = hilbert(buf_ifft)
        buf_ht = np.fft.fft(buf_ht_ifff)[:len(buf_ht) // 2 + 1]

        figure, axes = plt.subplots(nrows=2)
        axes[0].plot(20 * np.log10(np.abs(amp_target_linear)), label='before')
        axes[0].plot(20 * np.log10(np.abs(amp_target_linear_hilbert)),
                     label='rfft')
        if 'buf_ht' in locals():
            axes[0].plot(20 * np.log10(np.abs(buf_ht)), label='fft')

        axes[1].plot(np.angle(amp_target_linear, deg=True), label='before')
        axes[1].plot(np.angle(amp_target_linear_hilbert, deg=True),
                     label='rfft')
        if 'buf_ht' in locals():
            axes[1].plot(np.angle(buf_ht, deg=True), label='fft')

        for ax in axes:
            ax.set_xscale('log')
            ax.set_xlim(f_linear[0], f_linear[-1])
            ax.legend()

        del buf, buf_ht
    del amp_target_linear

    # 2-f. Resample in Logarithmic frequency
    target_amp = np.zeros_like(f_target_linear)
    target_freq = np.zeros_like(f_target_linear)
    target_amp = amp_target_linear_hilbert[index_log2linear]
    target_freq = f_linear[index_log2linear]
    del index_log2linear, amp_target_linear_hilbert, w_linear

    if DEBUG_PLOT:
        figure, axes = plt.subplots(nrows=2)
        axes[0].scatter(target_freq,
                        20 * np.log10(np.abs(target_amp)),
                        marker='.',
                        label='interploated')
        axes[0].scatter(fc,
                        target_gain,
                        color='r',
                        marker='*',
                        label="target gain")
        axes[1].plot(target_freq,
                     np.angle(target_amp, deg=True),
                     label='interploated phase')
        for ax in axes:
            ax.set_xscale('log')
            ax.set_xlim(f_linear[0], f_linear[-1])
            ax.legend()
            plt.show()
        pass

    assert target_freq[0] == f_linear[0], 'First frequency is not same'
    return target_freq, target_amp


def genPolePosition(sampling_freq: int, cutoff_freq: np.array):
    """Generate pole position from cutoff frequency
        theta = 2*pi*fk/fs, for k= 1, 2, ..., N
        pole = e^{delta_theta_k/2}*e^{+-j*theta_k}
        
        delta_theta_1 = theta_2 - theta_1,
        delta_theta_k = (theta_k+1 - theta_k-1)/2,
        delta_theta_N = theta_N - theta_N-1

        ak,1 = (-pk + pk_conj) = -2|p_k|cos(theta_k)
        ak,2 = |pk|^2
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

def genZeroPosition(sampling_freq: int, a_coeff: tuple, target_freq: np.array,\
                target_h: np.array):
    """Generate zero position from target frequency and h
        High-Precision Parallel Graphical Eqaulizer
            H(z) = d0+ \sum_{k=1}^{N} ( bk,0 + bk,1 z^-1 ) / (1 + a_k,1z^-1 + a_k,2z^-2)   
            h = Mp
            p = [b1,0, b1,1, ..., bK,0, bK,1, d0]^T
        M contains 1/(1 + ak,1 e^(-jwn) + ak,2 e^(-jwn)^2)) and e^(-jwn)/(1 + ak,1 e^(-jwn) + ak,2 e^(-jwn)^2))
        The last column of M belongs to the direct path gain d0, and thus all of its elements are one.
    """
    fs, fn, Hn = sampling_freq, target_freq, target_h
    a, z, wn = np.array(a_coeff), -1j, 2 * np.pi * fn / fs    # z = e^(-jwn) ?

    # (1, e^(-jwn), e^(-jwn)^2)
    denominator = np.array(
        [np.ones_like(wn), np.exp(z * wn),
         np.exp(2 * z * wn)])
    denominator = denominator.transpose()
    """TODO 1"""

    # M = 1/(1+ak,1*e^(-jwn)+ak,2*e^(-jwn)^2)
    M_coeff = 1 / np.matmul(denominator,
                            a)    # (wn, a_coeff) * (a_coeff, fc) = (wn, fc)

    # M = [1 + ak,1 e^(-jwn) + ak,2 e^(-jwn)^2)), e^(-jwn)/(1 + ak,1 e^(-jwn) + ak,2 e^(-jwn)^2)) , ... , 1]
    # (wn, fc) -> (wn, fc, 2) -> (wn, fc*2) -> append 0 -> (wn, fc*2+1) -> (wn, fc*2+1) * (fc*2+1, 1) = (wn, 1) ~= hn
    M_coeff = np.expand_dims(M_coeff, axis=-1)    # (wn, fc, 1)
    nominator = np.array([np.ones_like(wn), np.exp(z * wn)])    # (wn, 2)
    nominator = np.expand_dims(nominator.transpose(), axis=1)    # (wn, 1, 2)
    M_coeff = np.matmul(M_coeff,
                        nominator)    # (wn, fc, 1) * (wn, 1, 2) = (wn, fc, 2)
    M_coeff = M_coeff.reshape(
        (M_coeff.shape[0], M_coeff.shape[1] * M_coeff.shape[2]
        ))    # (wn, fc*2)    [b0,0, b0,1, b1,0, b1,1, b2,0, b2,1 ...]
    M_coeff = np.append(
        M_coeff, np.ones(shape=(M_coeff.shape[0], 1)),
        axis=-1)    # (wn, fc*2+1)  [b0,0, b0,1, b1,0, b1,1, b2,1, ..., 1]
    """TODO 2, 3"""

    # extract real part
    M_real = M_coeff.real
    hn_real = Hn.real
    # p_opt = M+ @ h_t, M+ = (MT @ M)^-1 @ MT
    W_wn = 1 / np.power(np.abs(hn_real), 2)
    M_real_T = W_wn * M_real.T    # Scaling - if not scaling, lowest part error is worst case
    M_plus_real = np.matmul(np.linalg.inv(np.matmul(M_real_T, M_real)),
                            M_real_T)
    p_opt_real = np.matmul(M_plus_real, hn_real)

    # # Complex computation
    # W_wn = 1/np.power(np.abs(Hn), 2)
    # print(W_wn.shape, M_coeff.shape)
    # M_H = W_wn*np.array(np.matrix(M_coeff).getH())
    # M_plus = np.matmul(np.linalg.inv(np.matmul(M_H, M_coeff)), M_H)
    # p_opt = np.matmul(M_plus, Hn)
    # Ht = np.matmul(M_coeff, p_opt)
    # Wn = 1/np.abs(Hn)
    # error = np.sum(np.power(np.abs(Hn - Ht), 2), axis=-1)

    # [TODO] extract real part
    # M_real = np.array([M_coeff.real, M_coeff.imag])
    # hn_real = np.array([Hn.real, Hn.imag])
    # print(M_real.shape, hn_real.shape)
    # W_wn = 1/np.power(np.abs(hn_real), 2)
    # M_real_T = M_real.transpose(0, 2, 1)    # Scaling - if not scaling, lowest part error is worst case
    # print(M_real.shape, M_real_T.shape)
    # print(np.matmul(M_real_T,  M_real).shape)
    # real_M, imag_M = np.matmul(M_real_T,  M_real)
    # print(real_M.shape, imag_M.shape)
    # print(imag_M)
    # real_M, imag_M = np.linalg.inv(real_M), np.linalg.inv(imag_M)
    # print(real_M.shape, imag_M.shape)
    # M_plus_real = np.matmul(np.linalg.inv(np.matmul(M_real_T,  M_real)), M_real_T)
    # p_opt_real = np.matmul(M_plus_real, hn_real)

    if DEBUG_PRINT:
        Ht = np.matmul(M_real, p_opt_real)
        error = np.sum(np.power(np.abs(Hn - Ht), 2), axis=-1)
        print(f'Error: {error}')

    if DEBUG_PLOT:
        Ht = np.matmul(M_real, p_opt_real)
        _, axes = plt.subplots(nrows=3)
        axes[0].scatter(target_freq,
                        20 * np.log10(np.abs(Ht)),
                        marker='.',
                        label='Calculated Gain')
        axes[0].scatter(target_freq,
                        20 * np.log10(np.abs(Hn)),
                        color='r',
                        marker='*',
                        label="Target gain")
        axes[1].plot(target_freq,
                     np.angle(Ht, deg=True),
                     label='Calculated phase')
        axes[1].plot(target_freq,
                     np.angle(Hn, deg=True),
                     color='r',
                     label='Target phase')
        axes[2].scatter(target_freq,
                        np.abs(20 * np.log10(np.abs(Ht)) -
                               20 * np.log10(np.abs(Hn))),
                        marker='.',
                        label='Difference Gain')

        for ax in axes:
            ax.set_xscale('log')
            ax.set_xlim(target_freq[0], target_freq[-1])
            # ax.legend()
        plt.show()

        # Complex computation
        # M_H = np.array(np.matrix(M).getH())
        # M_plus = np.matmul(np.linalg.inv(np.matmul(M_H, M)), M_H)
        # p_opt = np.matmul(M_plus, Hn)
        # Ht = np.matmul(M, p_opt)
        # Wn = 1/np.abs(Hn)
        # error = np.sum(np.power(np.abs(Hn - Ht), 2), axis=-1)

    del M_coeff, M_real, M_real_T, M_plus_real, hn_real, W_wn

    # Extract b_coff
    d, p_opt_real = p_opt_real[-1], p_opt_real[:-1]
    b0, b1 = p_opt_real[0::2], p_opt_real[1::2]

    assert b0[0] == p_opt_real[0] and b0.shape == b1.shape
    return d, (b0, b1, np.zeros_like(b0))


class GraphicalEqualizer(object):
    """Graphical Equalizer
        H(z) = d + sum_k=1^K (bk,0 + bk,1z-1) / (1 + ak,0z-1 + ak,1z-2)
    """

    def __init__(self, sampling_freq, cufoff_freq: np.array, gain: np.array,
                 Q: np.array) -> None:
        self.sampling_freq = sampling_freq
        self.cufoff_freq = cufoff_freq
        self.gain = gain
        self.steps = 2**15
        self.origin = 10
        self.axes = []
        self.freq_target, self.H_target = genTargetResponse(
            self.origin, self.sampling_freq, self.cufoff_freq, self.gain,
            self.steps)
        self.a = genPolePosition(self.sampling_freq, self.cufoff_freq)
        self.d, self.b = genZeroPosition(self.sampling_freq, self.a,
                                         self.freq_target, self.H_target)

    def write_to_file(self, filename):
        with open(filename, 'w') as f:
            for ch in range(len(self.b[0])):
                # f.write(f'{ch}: ')
                f.write(f'{self.b[0][ch]} ')
                f.write(f'{self.b[1][ch]} ')
                f.write(f'{self.b[2][ch]} ')
                f.write(f'{self.a[0][ch]} ')
                f.write(f'{self.a[1][ch]} ')
                f.write(f'{self.a[2][ch]}\n')
            f.write(f'{self.d}\n')
        print("Written to file")

    def freqz(self):
        """Apply parallel filter to data
                        jw                  -jw              -jwM
                jw   Bk(e  )    bk[0] + bk[1]e    + ... + bk[M]e
            H_k(e  ) = ------ = -----------------------------------
                        jw                  -jw              -jwN
                    Ak(e  )    ak[0] + ak[1]e    + ... + ak[N]e
            
                jwn                         jwn           
            H(e  ) =  bias +  sigma_k H_k(e  )
        """
        fs = self.sampling_freq
        fc_target, gain_target = self.freq_target, 20 * np.log10(self.H_target)
        num_step = self.steps
        bias = self.d
        w = np.linspace(0, np.pi, num_step, endpoint=True)
        zm = np.array([np.ones_like(w), np.exp(-1j * w), np.exp(-1j * 2 * w)])

        a_coeff = np.array(self.a).T
        b_coeff = np.array(self.b).T
        coeff = np.stack([b_coeff, a_coeff], axis=-1).transpose(0, 2, 1)
        h = coeff @ zm
        h = h.transpose(2, 0, 1)
        h = h[:, :, 0] / h[:, :, 1]
        h = np.sum(h, axis=-1)
        h = np.ones_like(h) * bias + h
        h = h.real

        plt.scatter(w / (2 * np.pi) * fs,
                    20 * np.log10(abs(h)),
                    marker='*',
                    color='r',
                    label='simulation')
        plt.scatter(fc_target,
                    gain_target,
                    marker='o',
                    color='k',
                    label='target')

        # x, y = plot_audacity_freq_response()
        # plt.scatter(x, y, marker='.', color='g', label='after processing')

        plt.xscale('log')
        plt.grid()
        plt.legend()

        # Plot including phase response
        # _, axes = plt.subplots(nrows=2)
        # axes[0].scatter(w/(2*np.pi) * fs, 20*np.log10(abs(h)), marker='*', color='r')
        # axes[0].scatter(fc, gain)
        # axes[1].scatter(w/(2*np.pi) * fs, np.angle(h, deg=True), marker='*', color='r')
        # for ax in axes:
        #     ax.set_xscale('log')
        #     ax.grid()
        #     # ax.set_xlim(10, fs/2)

        plt.show()

    def plot_debug(self):
        if len(self.axes) == 0:
            print('No axes to plot')


if '__main__' == __name__:
    import os
    root = os.getcwd()
    # sampling frequnecy
    fs = 48000

    # cuf-off freuqency case 1
    fc = np.array((20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
                   400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
                   5000, 6300, 8000, 10000, 12500, 16000, 20000))

    # cutoff frequency case 2
    fB = np.array([
        2.3, 2.9, 3.6, 4.6, 5.8, 7.3, 9.3, 11.6, 14.5, 18.5, 23.0, 28.9, 36.5,
        46.3, 57.9, 72.9, 92.6, 116, 145, 185, 232, 290, 365, 463, 579, 730,
        926, 1158, 1447, 1853, 2316
    ])
    fU = np.array([
        22.4, 28.2, 35.5, 44.7, 56.2, 70.8, 89.1, 112, 141, 178, 224, 282, 355,
        447, 562, 708, 891, 1120, 1410, 1780, 2240, 2820, 3550, 4470, 5620,
        7080, 8910, 11200, 14100, 17800, 22050
    ])
    fL = np.zeros_like(fU)
    fL[0] = 17.5
    fL[1:] = fU[:-1]

    fc_twice = np.zeros((2, len(fc)))
    fc_twice[0, :] = np.append(10, fU[:-1])
    fc_twice[1, :] = fc
    fc_twice = fc_twice.reshape((fc_twice.shape[0] * fc_twice.shape[1],),
                                order='F')

    # gain case 1
    gain_c1 = np.array([
        12, 12, 10, 8, 4, 1, 0.5, 0, 0, 6, 6, 12, 6, 6, -12, 12, -12, -12, -12,
        -12, 0, 0, 0, 0, -3, -6, -9, -12, 0, 0, 0
    ])

    # gain case 2

    gain_c2 = np.zeros_like(fc)
    gain_c2[0::2] = 12
    gain_c2[1::2] = -12

    # set the gain
    gain = gain_c1

    gain_twice = np.zeros((2, len(fc)))
    gain_twice[0, :] = gain
    gain_twice[1, :] = gain
    gain_twice = gain_twice.reshape(
        (gain_twice.shape[0] * gain_twice.shape[1],), order='F')
    fc_twice = fc_twice[1:]
    gain_twice = gain_twice[1:]

    # set the Q, not implemented yet
    Q = np.ones_like(fc)

    eq = GraphicalEqualizer(fs, fc, gain, Q)
    file = '/lib/data/test_graphic_eq_g1.txt'
    eq.write_to_file(f'{root}/{file}')

    # file = '/lib/data/test_graphic_eq_ftwice_g1.txt'
    # eq = GraphicalEqualizer(fs, fc_twice, gain_twice, Q)
    # eq.write_to_file('f'{root}/{file}')
