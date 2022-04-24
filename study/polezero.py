"""Reference.
    https://ccrma.stanford.edu/~jos/filters/Elementary_Filter_Sections.html
    This file follows the contents above link. 
"""
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

SAMPLING_FREQUENCY = 128

def freqz_scratch(b=None, a=None, worN=512, whole=False, include_nyquist=False):
    """
        Compute the frequency response of a digital filter.

        Given the M-order numerator `b` and N-order denominator `a` of a digital
        filter, compute its frequency response::

                    jw                 -jw              -jwM
            jw    B(e  )    b[0] + b[1]e    + ... + b[M]e
        H(e  ) = ------ = -----------------------------------
                    jw                 -jw              -jwN
                A(e  )    a[0] + a[1]e    + ... + a[N]e
    """
    
    N = worN
    lastpoint = 2 * np.pi if whole else np.pi
    w = np.linspace(0, lastpoint, N, endpoint=include_nyquist and not whole)
    zi = np.array([np.ones_like(w), np.exp(-1j*w), np.exp(-2j*w)], dtype=complex).T

    if b is None:
        b_poly = 1
    else:
        _b = np.zeros(3, dtype=complex)
        _b[:len(b)] = b
        b_poly = np.sum(_b*zi, axis=-1, dtype=complex)

    if a is None:
        a_poly = 1
    else:
        _a = np.zeros(3, dtype=complex)
        _a[:len(a)] = a 
        a_poly = np.sum(_a*zi, axis=-1, dtype=complex)
    
    # same process
    # from numpy.polynomial.polynomial import polyval
    # zm1 = np.exp(-1j * w)
    # h = (polyval(zm1, b, tensor=False) /
    #     polyval(zm1, a, tensor=False))

    return w, b_poly/a_poly


def test_freqz_scratch():
    """Test freqz_scratch
    """
    b0 = 1
    b1 = 1

    b_coeff = np.array([b0, b1])
    a_coeff = None

    w_scratch, H_scratch = freqz_scratch(b=b_coeff, worN=SAMPLING_FREQUENCY)
    w, H = signal.freqz(b=b_coeff, worN=SAMPLING_FREQUENCY)

    print(np.allclose(w, w_scratch))
    print(np.allclose(H, H_scratch))

    _fig = plt.figure(figsize=(10, 10))

    ax = plt.Axes()

    ax = [0]*4
    ax[0] = _fig.add_subplot(221)
    ax[1] = _fig.add_subplot(222)
    ax[2] = _fig.add_subplot(223)
    ax[3] = _fig.add_subplot(224)

    ax[0].plot(w_scratch, np.abs(H_scratch), '.')
    ax[0].set_title('Scratch')
    ax[0].set_xlabel('Frequency [rad/sample]')
    ax[0].set_ylabel('Magnitude')
    ax[0].grid(True)

    ax[1].plot(w, np.abs(H), '.')
    ax[1].set_title('Scipy')
    ax[1].set_xlabel('Frequency [rad/sample]')
    ax[1].set_ylabel('Magnitude')
    ax[1].grid(True)

    ax[2].plot(w_scratch, np.angle(H_scratch), '.')
    ax[2].set_title('Scratch')
    ax[2].set_xlabel('Frequency [rad/sample]')
    ax[2].set_ylabel('Phase [rad]')
    ax[2].grid(True)

    ax[3].plot(w, np.angle(H), '.')
    ax[3].set_title('Scipy')
    ax[3].set_xlabel('Frequency [rad/sample]')
    ax[3].grid(True)

    plt.show()


def one_zero():
    """
    Difference equation : y[n] = b0*x[n] + b1*x[n-1]
    z transform         : Y[z] = b0*X[z] + b1*z^{-1}*X[z]
    Transfer function   : H(z) = b0 + b1*z^{-1}
    Frequency response  : H(e^{jwT}) = b0 + b1*e^{-jwT}
    """
    b0 = 1
    b1_list = np.arange(-1, 1.1, 0.1)


    _fig = plt.figure(figsize=(10, 12))
    ax = [0]*2
    ax[0] = _fig.add_subplot(211)
    ax[1] = _fig.add_subplot(212)


    for b1 in b1_list:
        b_coeff = np.array([b0, b1])
        a_coeff = None
        w, H = signal.freqz(b=b_coeff, worN=SAMPLING_FREQUENCY)

        ax[0].plot(w, np.abs(H), '.', label=f"b1={b1}")
        ax[0].set_title('Magnitude')
        ax[0].set_xlabel('Normalized Frequency [rad/sample]')
        ax[0].set_ylabel('Magnitude')
        ax[0].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        ax[0].grid(True)

        ax[1].plot(w, np.angle(H), '.', label=f"b1={b1}")
        ax[1].set_title('Phase')
        ax[1].set_xlabel('Normalized Frequency [rad/sample]')
        ax[1].set_ylabel('Phase [rad]')
        ax[1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def one_pole():
    """
    Difference equation : y[n] = b0*x[n] - a1*y[n-1]
    z transform         : Y[z] = b0*X[z] - a1*z^{-1}Y[z]
    Transfer function   : H(z) = b0 / (1+a1*z^{-1})
    Frequency response  : H(e^{jwT}) =b0 / (1+a1*e^{-jwT})
    """
    b0 = 1
    
    a0 = 1
    a1 = 0.9
    a1_list = np.array([-0.9, -0.8, -0.6, 0, 0.6, 0.8, 0.9])
    _fig = plt.figure(figsize=(10, 12))
    ax = [0]*2
    ax[0] = _fig.add_subplot(211)
    ax[1] = _fig.add_subplot(212)

    for a1 in a1_list:
        b_coeff = np.array([b0])
        a_coeff = np.array([a0, a1])
        w, H = signal.freqz(b=b_coeff,a=a_coeff, worN=SAMPLING_FREQUENCY)

        ax[0].plot(w, np.abs(H), '.', label=f"a1={a1}")
        ax[0].set_title('Magnitude')
        ax[0].set_xlabel('Normalized Frequency [rad/sample]')
        ax[0].set_ylabel('Magnitude')
        ax[0].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        ax[0].grid(True)

        ax[1].plot(w, np.angle(H), '.', label=f"a1={a1}")
        ax[1].set_title('Phase')
        ax[1].set_xlabel('Normalized Frequency [rad/sample]')
        ax[1].set_ylabel('Phase [rad]')
        ax[1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def two_pole():
    """
    Difference equation : y[n] = b0*x[n] - a1*y[n-1] - a2*y[n-1]
    z transform         : Y[z] = b0*X[z] - a1*z^{-1}Y[z] - a2*z^{-2}Y[z]
    Transfer function   : H(z) = b0 / (1+a1*z^{-1}+a2*z^{-2})
    Frequency response  : H(e^{jwT}) =b0 / (1+a1*e^{-jwT}+a2*e^{-j2wT})

    * If pole is complex,
    p_1 = Re^{j\theta_c}, p_2 = Re^{-j\theta_c},
    a_1 = -2Rcos(\theta_c), a_2 = -R^2

    y(n) = x(x) + 2Rcos(\theta_c)y(n-1)-R^2 y(n-2),
    H(z) = b0 / (1-2Rcos(\theta_c)z^{-1}+R^2 z^{-2})
    """
    R_list = np.array([0, 0.3, 0.6, 0.8, 0.9])
    theta_c = np.pi/4

    _fig = plt.figure(figsize=(10, 12))
    ax = [0]*2
    ax[0] = _fig.add_subplot(211)
    ax[1] = _fig.add_subplot(212)

    for R in R_list:
        b0 = 1
        b_coeff = np.array([b0])
        a_coeff = np.array([1, -2*R*np.cos(theta_c), R**2])

        w, H = signal.freqz(b=b_coeff,a=a_coeff, worN=SAMPLING_FREQUENCY)

        ax[0].plot(w, np.abs(H), '.', label=f"R={R}")
        ax[0].set_title('Magnitude')
        ax[0].set_xlabel('Normalized Frequency [rad/sample]')
        ax[0].set_ylabel('Magnitude')
        ax[0].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        ax[0].grid(True)

        ax[1].plot(w, np.angle(H), '.', label=f"R={R}")
        ax[1].set_title('Phase')
        ax[1].set_xlabel('Normalized Frequency [rad/sample]')
        ax[1].set_ylabel('Phase [rad]')
        ax[1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def two_zero():
    """
    Difference equation : y[n] = b0*x[n] + b1*x[n-1] + b2+y[n-1]
    z transform         : Y[z] = b0*X[z] + b1*z^{-1}X[z] + b2*z^{-2}Y[z]
    Transfer function   : H(z) = b0+b1*z^{-1}+b2*z^{-2}
    Frequency response  : H(e^{jwT}) =b0+b1*e^{-jwT}+b2*e^{-j2wT})

    * If pole is complex,
    p_1 = Re^{j\theta_c}, p_2 = Re^{-j\theta_c},
    b_1/b_0 = -2Rcos(\theta_c), b_2/b_0 = R^2

    y(n) = b_0{x(n) - 2Rcos(\theta_c)x(n-1)+R^2 x(n-2)},
    H(z) = b0[x(n)-2Rcos(\theta_c)z^{-1}+R^2 z^{-2}]
    """
    R_list = np.array([0, 0.3, 0.6, 0.8, 0.9])
    theta_c = np.pi/4

    _fig = plt.figure(figsize=(10, 12))
    ax = [0]*2
    ax[0] = _fig.add_subplot(211)
    ax[1] = _fig.add_subplot(212)

    for R in R_list:
        b0 = 1
        b_coeff = np.array([b0, -b0*2*R*np.cos(theta_c), b0*R**2])
        a_coeff = np.array([1])

        w, H = signal.freqz(b=b_coeff,a=a_coeff, worN=SAMPLING_FREQUENCY)

        ax[0].plot(w, np.abs(H), '.', label=f"R={R}")
        ax[0].set_title('Magnitude')
        ax[0].set_xlabel('Normalized Frequency [rad/sample]')
        ax[0].set_ylabel('Magnitude')
        ax[0].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        ax[0].grid(True)

        ax[1].plot(w, np.angle(H), '.', label=f"R={R}")
        ax[1].set_title('Phase')
        ax[1].set_xlabel('Normalized Frequency [rad/sample]')
        ax[1].set_ylabel('Phase [rad]')
        ax[1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def complex_resonator():
    """
    H(z) = g_1/(1-p*z^{-1}) + g_2/(1-p_conj*z^{-1})
    """
    R = 0.8
    theta_c = np.pi/8
    frame_size = SAMPLING_FREQUENCY
    b0 = 1


    # each one pole
    p = R*np.exp(1j*theta_c)
    b_coeff = np.array([b0], dtype=complex)
    a_coeff = np.array([1, -p], dtype=complex)
    w, H_one_pole = signal.freqz(b=b_coeff,a=a_coeff, worN=SAMPLING_FREQUENCY, whole=True)
    H_one_pole = np.roll(H_one_pole, frame_size//2)     
    w -= np.pi
    
    p_conj = R*np.exp(-1j*theta_c)
    b_coeff = np.array([b0], dtype=complex)
    a_coeff = np.array([1, -p_conj], dtype=complex)
    _, H_one_pole_conj = signal.freqz(b=b_coeff,a=a_coeff, worN=SAMPLING_FREQUENCY, whole=True)
    H_one_pole_conj = np.roll(H_one_pole_conj, frame_size//2)     

    # two pole
    # H(z) = g_1/(1-p*z^{-1}) + g_2/(1-p_conj*z^{-1})
    H = H_one_pole + H_one_pole_conj

    _fig = plt.figure(figsize=(10, 12))
    ax = [0]*2
    ax[0] = _fig.add_subplot(211)
    ax[1] = _fig.add_subplot(212)  

    ax[0].plot(w, np.abs(H), '-', label=f"R={R}, theta_c=pi/{int(np.pi/theta_c)}  Two poles")
    ax[0].plot(w, np.abs(H_one_pole), '.', label=f"R={R}, theta_c=pi/{int(np.pi/theta_c)}  One poles")
    ax[0].plot(w, np.abs(H_one_pole_conj), '.', label=f"R={R}, theta_c=pi/{int(np.pi/theta_c)}  One poles with conjugate")
    ax[0].set_title('Magnitude')
    ax[0].set_xlabel('Normalized Frequency [rad/sample]')
    ax[0].set_ylabel('Magnitude')
    ax[0].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax[0].grid(True)

    ax[1].plot(w, np.angle(H), '-', label=f"R={R}, theta_c=pi/{int(np.pi/theta_c)}  Two poles")
    ax[1].plot(w, np.angle(H_one_pole), '.', label=f"R={R}, theta_c=pi/{int(np.pi/theta_c)}  One poles")
    ax[1].plot(w, np.angle(H_one_pole_conj), '.', label=f"R={R}, theta_c=pi/{int(np.pi/theta_c)}  One poles with conjugate")

    ax[1].set_title('Phase')
    ax[1].set_xlabel('Normalized Frequency [rad/sample]')
    ax[1].set_ylabel('Phase [rad]')
    ax[1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    # test_freqz_scratch()
    # one_zero()
    # one_pole()
    # two_pole()
    # two_zero()
    # complex_resonator()
    pass
