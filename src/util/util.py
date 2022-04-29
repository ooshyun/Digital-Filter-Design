import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile as wav
from scipy.fftpack import *
from scipy.signal import tf2zpk
from matplotlib.ticker import MultipleLocator


def plot_pole_zero_analysis(ax: plt.axes, b, a):
    """Plot poles and zeros through filter coefficient
    """
    if isinstance(a[0], np.ndarray):
        len_zi = len(a[0])
    else:
        len_zi = 1

    for i_zi in range(len_zi):
        if len_zi == 1:
            zpk = tf2zpk(b, a)
        else:
            zpk = tf2zpk(
                (b[0][i_zi], b[1][i_zi], b[2][i_zi]),
                (a[0][i_zi], a[1][i_zi], a[2][i_zi]),
            )
        kw_z = dict(c="C0", marker="o", ms=9, ls="none", mew=1, mfc="none", alpha=1)
        kw_p = dict(c="k", marker="x", ms=9, ls="none", mew=1)
        z, p, _ = zpk
        ax.plot(np.real(p), np.imag(p), **kw_p)
        ax.plot(np.real(z), np.imag(z), **kw_z)

        kw_artist = dict(edgecolor="gray", linestyle="-", linewidth=1)
        circle = plt.Circle(xy=(0, 0), radius=1, facecolor="none", **kw_artist)
        ax.add_artist(circle)
    ax.axis([-1.1, 1.1, -1.1, 1.1])
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_ylabel(r"$\Im (z)$")
    ax.set_xlabel(r"$\Re (z)$")

    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))


def plot_frequency_response(
    ax: plt.axes,
    x,
    y,
    xlim,
    xlabels,
    xaxis=None,
    xtitle=None,
    yaxis=None,
    ylim=None,
    ylabels=None,
    ytitle=None,
):
    """Plot x and y data based on frequency scaling 
    """
    # ax.semilogx(x, y)
    ax.plot(x, y)
    ax.set_xscale("log")
    ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if xaxis is not None:
        ax.set_xticks(xaxis)
    ax.set_xticklabels(xlabels)
    ax.minorticks_off()
    ax.grid(True)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)


def cvt_float2fixed(coeff):
    """Print fixed-point coefficients for C/C++.
    """
    from .fi import fi

    coeff = np.array(coeff).tolist()

    for location in range(len(coeff)):
        for i in range(len(coeff[location])):
            coeff[location][i] = fi(coeff[location][i], 1, 32, 30, 1, "Hex")

    for location in range(len(coeff)):
        print("{", end="")
        for i in range(len(coeff[location])):
            print(coeff[location][i], end="")
            if i == len(coeff[location]) - 1:
                continue
            else:
                print(", ", end="")
        print("}")


def cvt_char2num(array: list):
    """Convert char array including the meaning of int
        to int array.
    """
    for i in range(len(array)):
        if isinstance(array[i], list):
            cvt_char2num(array[i])
        else:
            if isinstance(array[i], str):
                if "j" in array[i]:
                    array[i] = complex(array[i])
                else:
                    array[i] = float(array[i])
            elif isinstance(array[i], float):
                pass
            elif isinstance(array[i], int):
                pass
            else:
                raise ValueError("Wrong type")


def cvt_pcm2wav(from_file, to_file, sample_rate, dtype):
    """[TODO] Convert PCM file to WAV file.
    """
    raise NotImplementedError
    with open(from_file, "rb") as opened_pcm_file:
        buf = opened_pcm_file.read()
        pcm_data = np.frombuffer(buf, dtype=dtype)
        wav_data = librosa.util.buf_to_float(pcm_data, 2)
    wav.write(to_file, sample_rate, wav_data)
