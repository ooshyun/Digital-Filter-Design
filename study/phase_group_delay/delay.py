"""
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as intep
import scipy.signal as signal

def swanal(t, f, fs, amp, phase, B, A):
    """sine wave analysis
        Reference. https://ccrma.stanford.edu/~jos/fp/Sine_Wave_Analysis.html
    """
    if not isinstance(B, list):
        len_b = 1
    else:
        len_b = len(B)

    if not isinstance(A, list):
        len_a = 1
    else:
        len_a = len(A)
    len_delay = max(len_b, len_a)

    ampin = amp
    phasein = phase

    yin = np.zeros(shape=(len(f), len(t)))
    yout = np.zeros(shape=(len(f), len(t)))
    gain = np.zeros_like(f)
    phase = np.zeros_like(f)

    for k, fk in enumerate(f):
        # wn = 2 * np.pi * fk / fs
        s = ampin * np.cos(2 * np.pi * fk * t + phasein)
        yn = signal.lfilter(B, A, s)
        yin[k, :] = s
        yout[k, :] = yn

    return yin, yout, gain, phase


def real_sine_wave_analysis(flag_plot=False):
    """Real Sine-wave anaylsis
        Reference. https://ccrma.stanford.edu/~jos/fp/Sine_Wave_Analysis.html
        1. Use Equation

        2. Use filter cofficient b, a
    """
    # 1. Using equation f(y) = g(x)
    sampling_rate = 44100
    dt = 1 / sampling_rate

    step_course = 1
    step_fine = step_course / 100
    n_sample_min = 0
    n_sample_max = 20
    n_sample = np.arange(n_sample_min, n_sample_max + 1, step_course)
    n_sample_fine = np.arange(n_sample_min, n_sample_max, step_fine)
    wn = 2 * np.pi * sampling_rate / 4 * n_sample * dt % (2 * np.pi)

    amp_x = 1
    phi_x = np.pi / 2
    xn = amp_x * np.cos(2 * np.pi * sampling_rate / 4 * n_sample * dt + phi_x)
    f_xn_intep = intep.interp1d(n_sample * dt, xn, kind="cubic")

    yn = np.zeros_like(xn)
    bias = 0
    yn[0] = xn[0] + bias
    for n in range(1, len(xn)):
        yn[n] = xn[n] + xn[n - 1]
    f_yn_intep = intep.interp1d(n_sample * dt, yn, kind="cubic")

    if flag_plot:
        plt.plot(n_sample, xn, "o", color="r")
        plt.plot(n_sample_fine, f_xn_intep(n_sample_fine * dt), linestyle="dashed")

        plt.plot(n_sample, wn, ".", color="k")

        plt.plot(n_sample, yn, "x", color="b", mew=3)
        plt.plot(n_sample_fine, f_yn_intep(n_sample_fine * dt), linestyle="dashed")

        plt.xlim(0, 20)
        plt.xticks(np.arange(0, 25, 5))
        plt.grid()
        plt.show()

    # 2. Using cofficient
    EPS = 10e-5
    B = [1, 1]      # filter feedforward coefficients y[n] = x[n] + x[n-1]
    A = 1           # filter feedback coefficients (none)
    N = 10          # number of sinusoidal test frequencies

    fs = 1          # sampling rate in Hz (arbitrary)
    T = 1 / fs      # sampling interval in seconds
    fmax = fs / 2   # highest frequency to look at
    df = fmax / (N - 1)  # spacing between frequencies
    f = np.linspace(0, fmax, N, endpoint=True)  # sampled frequency axis
    tmax = 1 / df  # 1 cycle at lowest non-dc freq
    t = np.linspace(0, tmax, int(tmax // T) + 1, endpoint=True)  # sampled time axis

    ampin = 1
    phasein = 0
    y_in, y_out, gain, phase = swanal(
        t, f, fs, ampin, phasein, B, A
    )  # sinewave analysis in matlab

    ntransient = len(B) - 1
    yss = y_in[5]
    f_intep = intep.interp1d(t, yss, kind="cubic")
    t_fine = np.linspace(0, tmax, (int(tmax // T) + 1) * 100, endpoint=True)

    if flag_plot:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

        ax[0][0].plot(t, yss, "*")
        ax[0][0].plot(t_fine, f_intep(t_fine), linestyle="dashed")
        ax[0][0].set_title("Filter Input Sinusoid, f(6)=0.28")
        ax[0][0].set_xlabel("Time (sec)")
        ax[0][0].set_ylabel("Amplitude", rotation=90)
        ax[0][0].set_xlim(0, 10)
        ax[0][0].set_xticks(np.arange(0, 11, 1))
        ax[0][0].set_yticks(np.arange(-1.5, 2.0, 0.5))
        ax[0][0].grid()

        yss = y_out[5]
        ax[1][0].plot(t, yss, "*")
        ax[1][0].set_xlim(0, 10)
        ax[1][0].set_title("Filter Input Sinusoid, f(6)=0.28")
        ax[1][0].set_xlabel("Time (sec)")
        ax[1][0].set_ylabel("Amplitude", rotation=90)
        ax[1][0].set_xticks(np.arange(0, 11, 1))
        ax[1][0].set_yticks(np.arange(-1.5, 2.0, 0.5))
        ax[1][0].grid()

        # the one-sample start-up transient is removed from the filter output signal y
        # to form the ``cropped'' signal yss (y steady state)
        yss = y_out[:, ntransient:]
        ampout = np.max(np.abs(yss), axis=-1)
        ampout[np.where(ampout == 0)] = EPS  # avoid divide by zero
        peakloc = np.zeros_like(ampout)
        for k, yk in enumerate(yss):
            peakloc[k] = np.where(yk == np.max(yk))[0][0]

        gains = ampout / ampin

        """TODO: 
            Phase Difference
            phaseDiff =  acos(min(1,max(-1,yss(end)/ampOut)))
                    - acos(min(1,max(-1,sss(end)/ampIn)));
        """
        sss = y_in[:, ntransient:]

        G_f = 2 * np.cos(np.pi * f / fs)
        Phi_f = -np.pi * f / fs / (2 * np.pi)

        f_fine = np.linspace(min(f), max(f), len(f) * 100)
        ax[0][1].plot(f, gains, "*", color="red")
        ax[0][1].plot(f_fine, intep.interp1d(f, G_f, kind="cubic")(f_fine))
        ax[0][1].set_title("Amplitude Response")
        ax[0][1].set_xlabel("Frequency (Hz)")
        ax[0][1].set_ylabel("Gain", rotation=90)
        ax[0][1].set_xlim(0, 0.5)
        ax[0][1].set_xticks(np.arange(0, 0.6, 0.1))
        ax[0][1].set_yticks(np.arange(0, 2.5, 0.5))
        ax[0][1].grid()

        ax[1][1].plot(f, Phi_f)
        ax[1][1].set_title("Phase Response")
        ax[1][1].set_xlabel("Frequency (Hz)")
        ax[1][1].set_ylabel("Phase Shift (cycles)", rotation=90)
        ax[1][1].grid()

        plt.show()


def complex_sine_wave_anaylsis(flag_plot=False):
    """Complex Sinusoid wave anaylsis
    fs: sampling rate
    H(e^{jwT}) = 1+e^{-jwT}
               = 2cos(wT/2)e^{-jwT/2}
    G(w) = 2 cos(wT/2)
    Phi(w) = -wT/2 = -pi * fn/fs
    
    function [gains, phases] = swanalc(t,f,B,A)
    SWANALC - Perform COMPLEX sine-wave analysis on the 
            digital filter having transfer function 
            H(z) = B(z)/A(z)

    Reference. https://ccrma.stanford.edu/~jos/fp/Complex_Sine_Wave_Analysis.html
    """
    B = [1, 1]  # filter feedforward coefficients y[n] = x[n] + x[n-1]
    A = [1]  # filter feedback coefficients (none)
    N = 10  # number of sinusoidal test frequencies

    fs = 1  # sampling rate in Hz (arbitrary)
    T = 1 / fs  # sampling interval in seconds
    fmax = fs / 2  # highest frequency to look at
    df = fmax / (N - 1)  # spacing between frequencies
    f = np.linspace(0, fmax, N, endpoint=True)  # sampled frequency axis
    t = np.linspace(0, 10, 10 - 0 + 1, endpoint=True)

    ampin = 1  # input signal amplitude
    phasein = 0  # input signal phase
    N = len(f)  # number of test frequencies
    gains = np.zeros(shape=(N,))  # pre-allocate amp-response array
    phases = np.zeros(shape=(N,))  # pre-allocate phase-response array

    ntransient = len(B) - 1

    if len(A) == 1 and ntransient == len(B) - 1:
        pass
    else:
        raise ValueError("Need to set transient response duration here")

    # average the results to minimize noise due to round-off error:
    for k in range(len(f)):  # loop over analysis frequencies
        s = ampin * np.exp(1j * 2 * np.pi * f[k] * t + phasein)  # test sinusoid
        y = signal.lfilter(B, A, s)  # run it through the filter

        yss = y[ntransient:]  # chop off transient
        ampout = np.mean(np.abs(yss))  # avg instantaneous amplitude
        gains[k] = ampout / ampin  # amplitude response sample
        sss = s[ntransient:]  # align with yss
        phases[k] = np.mean(np.angle(yss * np.conj(sss))) / (2 * np.pi)

    G_f = 2 * np.cos(np.pi * f / fs)
    Phi_f = -np.pi * f / fs / (2 * np.pi)

    if flag_plot:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

        f_fine = np.linspace(0, f[-1], len(f) * 100, endpoint=True)
        ax[0].plot(f_fine, intep.interp1d(f, G_f, kind="cubic")(f_fine), color="black")
        ax[0].plot(f, gains, "*", color="red")
        ax[0].set_xticks(np.arange(0, fmax + 0.05, 0.05))
        ax[0].set_yticks(np.arange(0, 2.5, 0.5))
        ax[0].grid(True, linestyle="dashed")
        ax[0].set_title("Amplitude Response")
        ax[0].set_xlabel("Frequency (Hz)")
        ax[0].set_ylabel("Gain", rotation=90)

        ax[1].plot(f, Phi_f, color="black")
        ax[1].plot(f, phases, "*", color="red")
        ax[1].set_xticks(np.arange(0, fmax + 0.05, 0.05))
        ax[1].set_yticks(np.arange(-0.25, 0.05, 0.05))
        ax[1].grid(True, linestyle="dashed")
        ax[1].set_title("Phase Response")
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_ylabel("Phase Shift (cycles)", rotation=90)
        plt.show()


def phase_unwrapping(flag_plot=False):
    """Plot ths simulation for unwrapping effect
        - unwrap
        - contract unit circle to 0.95
    """
    b, a = signal.ellip(4, 1, 20, 0.5)  # design lowpass filter
    b_contract = b.copy()
    b_contract *= 0.95 ** np.arange(1, len(b_contract) + 1)  # contract zeros by 0.95

    w, h = signal.freqz(b, a)
    w_contract, h_contract = signal.freqz(b_contract, a)  # compute frequency response

    theta = np.angle(h)
    theta_contract = np.angle(h_contract)  # unwrap phase

    if flag_plot:
        fig = plt.figure(figsize=(8, 6))
        axes = [0] * 6

        axes[0] = fig.add_subplot(221)  # 2 rows, 1 column, 1st subplot
        axes[1] = fig.add_subplot(245)  # 2 rows, 4 columns, 5th subplot
        axes[2] = fig.add_subplot(246)  # 2 rows, 4 columns, 6th subplot

        axes[3] = fig.add_subplot(222)  # 1 row, 2 columns, 2st subplot
        axes[4] = fig.add_subplot(247)  # 2 rows, 2 columns, 7th subplot
        axes[5] = fig.add_subplot(248)  # 2 rows, 2 columns, 8th subplot

        axes[0].plot(w_contract, np.abs(h_contract))  # plot magnitude
        axes[1].plot(w_contract, theta_contract)
        axes[2].plot(w_contract, np.unwrap(theta_contract))

        axes[0].set_title("Figure 1.1 Amplitude Response")
        axes[0].set_xlabel("Frequency (rad/sample)")
        axes[0].set_ylabel("Amplitude (dB)")
        axes[0].grid(True)

        for iax, ax in enumerate(axes[1:3]):
            ax.set_xticks(np.arange(0, 4, 0.5))
            ax.set_title(f"Figure 1.{iax+2} Phase Response")
            ax.set_xlabel("Frequency (rad/sample)")
            ax.set_ylabel("Phase (rad)")
            ax.grid(True)

        axes[3].plot(w, np.abs(h))  # plot magnitude
        axes[4].plot(w, theta)
        axes[5].plot(w, np.unwrap(theta))

        axes[3].set_title("Figure 2.1 Amplitude Response")
        axes[3].set_xlabel("Frequency (rad/sample)")
        axes[3].set_ylabel("Amplitude (dB)")
        axes[3].grid(True)

        for iax, ax in enumerate(axes[4:6]):
            ax.set_xticks(np.arange(0, 4, 0.5))
            ax.set_title(f"Figure 2.{iax+2} Phase Response")
            ax.set_xlabel("Frequency (rad/sample)")
            ax.set_ylabel("Phase (rad)")
            ax.grid(True)

        fig.tight_layout()
        plt.show()


def group_delay(flag_plot=False):
    """Plot the group delay of several filter
        - filter: butterworth, chebyshev 1, chebyshev 2, elliptic
    """
    Bb, Ab = signal.butter(4, 0.5)
    # order 4, cutoff at 0.5 * pi
    _, Hb = signal.freqz(Bb, Ab)
    _, Db = signal.group_delay((Bb, Ab))

    Bc1, Ac1 = signal.cheby1(4, 1, 0.5)
    # 1 dB passband ripple
    _, Hc1 = signal.freqz(Bc1, Ac1)
    _, Dc1 = signal.group_delay((Bc1, Ac1))

    Bc2, Ac2 = signal.cheby2(4, 20, 0.5)
    # 20 dB stopband attenuation
    _, Hc2 = signal.freqz(Bc2, Ac2)
    _, Dc2 = signal.group_delay((Bc2, Ac2))

    Be, Ae = signal.ellip(4, 1, 20, 0.5)
    # like cheby1 + cheby2
    _, He = signal.freqz(Be, Ae)
    w, De = signal.group_delay((Be, Ae))

    if flag_plot:
        fig = plt.figure(figsize=(8, 6))
        axes = [0, 0]
        axes[0] = fig.add_subplot(121)
        axes[1] = fig.add_subplot(122)

        for amplitude in [Hb, Hc1, Hc2, He]:
            axes[0].plot(w, np.abs(amplitude))

        for delay in [Db, Dc1, Dc2, De]:
            axes[1].plot(w, delay)

        axes[0].set_xlabel("Frequency (rad/sample)")
        axes[0].set_ylabel("Gain")
        axes[0].legend(["butter", "cheby1", "cheby2", "ellip"])
        axes[0].grid(True)

        axes[1].set_xlabel("Frequency (rad/sample)")
        axes[1].set_ylabel("Delay (samples)")
        axes[1].legend(["butter", "cheby1", "cheby2", "ellip"])
        axes[1].grid(True)

        plt.show()

if __name__=='__main__':
    """ Phase and Group Delay"""
    real_sine_wave_analysis(flag_plot=False)
    complex_sine_wave_anaylsis(flag_plot=False)
    phase_unwrapping(flag_plot=False)
    group_delay(flag_plot=False)