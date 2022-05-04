import numpy as np
import scipy
import matplotlib.pyplot as plt

"""
axes = fig.subplots(ncols, nrows)
axes[:nrows][:ncols]

                         ncols
             <---------------------------
          -   ----------------------------- 
          |   |             |
          |   |             |
        n |   -----------------------------
        r |   |             |
        o |   |             |
        w |   -----------------------------
        s |   |             |
          |   |             |

"""


def plot_freq_response_tf(transfer: tuple, fs, frame_size, xlim: tuple):
    if len(transfer) != 2:
        print(f"Wrong tf Format")
        return
    b, a = transfer
    w, h = scipy.signal.freqz(b, a, 1024)
    freq = w * fs * 1.0 / (2 * np.pi)
    plt.plot(freq, 20 * np.log10(abs(h)), "b")

    plt.xscale("log")
    plt.title(f"shelving filter, Frequency Response")
    if xlim:
        plt.xlim(xlim)
    plt.xticks(
        [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
        ["10", "50", "100", "200", "500", "1K", "2K", "5K", "10K", "20K"],
    )
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.yticks(range(-24, 25, 3))
    plt.margins(0, 0.1)
    plt.grid(True, color="0.7", linestyle="-", which="major", axis="both")
    plt.grid(True, color="0.9", linestyle="-", which="minor", axis="both")

    plt.show()


def transfer_response(transfer_func, size, logscale=False, half=False, shift=False):
    """ Transfer function Response """
    x = np.arange(size)
    w, h = scipy.signal.freqz(transfer_func, 1, worN=size, whole=True)
    """ 
        Amplitude equation
        amplitude = np.abs(h)
        amplitude(dB) = 20 * np.log10(np.abs(h))
    """
    amplitude = np.abs(h)

    # plt.plot(transfer_func,"*--", label='impulse')
    # plt.legend(loc='lower center')
    # plt.show()

    if logscale:
        if amplitude[0] == 0:
            amplitude = 0
        else:
            amplitude = 20 * np.log10(amplitude / amplitude[0])

    """
        Angle equation,  
        angle = 180 * np.angle(h) / np.pi  # Convert to degrees 
    """
    angle = np.angle(h, deg=True)

    xShift = np.zeros(size + 1)
    middle = int(len(x) / 2)

    xShift[middle:] = x[: middle + 1]
    xShift[:middle] = np.flip(x[1 : middle + 1]) * (-1)

    amplitudeShift = np.zeros(len(amplitude) + 1)
    amplitudeShift[middle:] = amplitude[: middle + 1]
    amplitudeShift[:middle] = np.flip(amplitude[1 : middle + 1])

    angleShift = np.zeros(len(angle) + 1)
    angleShift[middle:] = angle[: middle + 1]
    angleShift[:middle] = np.flip(angle[1 : middle + 1])

    """
                                  Symmetric
                 <------------------->|<-------------------->
                |--------------------------------------------
    Except    Bias                 Nyquist 
                         
    If frame size is 256[0:255], we got the 256[0:255] fft results,
    0: bias
    1-127: Symmetric(numbers: 127)
    128: Nyquist Frequency
    129-255: Symmetric(numbers: 127)                    
    """

    if half:
        size_half = int(size / 2)
        x = x[:size_half]

        amplitude = amplitude[:size_half]
        angle = angle[:size_half]

        xShift = xShift[size_half:]
        amplitudeShift = amplitudeShift[size_half:]
        angleShift = angleShift[size_half:]

    if shift:
        return xShift, amplitudeShift, angleShift
    else:
        return x, amplitude, angle


def plot_tranfer_response(xaxis, **kwargs):
    numPlot = kwargs["numbers"]

    lenX = kwargs["xlen"]
    if lenX == None:
        origin = None
        endX = None
    else:
        origin = int(len(xaxis) / 2)
        if lenX == -1:
            endX = -1
        else:
            endX = origin + lenX

    fig = plt.figure(constrained_layout=True, figsize=(24, 8))
    fig.set_facecolor("0.5")
    plotStyle = kwargs["plotStyle"]

    plot_list = [kwargs["amplitude"], kwargs["phase"], kwargs["object"]]

    if numPlot == 1:
        axes = fig.subplots(ncols=1, nrows=2)
        axes[0].plot(
            xaxis, plot_list[0][origin:endX], plotStyle, label=plot_list[-1].name
        )
        axes[1].plot(
            xaxis, plot_list[1][origin:endX], plotStyle, label=plot_list[-1].name
        )

    else:
        axes = fig.subplots(ncols=numPlot, nrows=2)
        for index_row, axes_rows in enumerate(axes):
            for index_col, ax in enumerate(axes_rows):
                ax.plot(
                    xaxis[origin:endX],
                    plot_list[index_row][index_col][origin:endX],
                    plotStyle,
                    label=plot_list[-1][index_col].name,
                )
                ax.grid()
                ax.legend(loc="lower center")

    plt.minorticks_on()
    plt.show()


def plot_time_domain(xaxis, **kwargs):
    numPlot = kwargs["numbers"]

    lenX = kwargs["xlen"]
    if lenX == None:
        origin = None
        endX = None
    else:
        origin = int(len(xaxis) / 2)
        if lenX == -1:
            endX = -1
        else:
            endX = origin + lenX

    fig = plt.figure(constrained_layout=True, figsize=(24, 8))
    fig.set_facecolor("0.5")
    plotStyle = kwargs["plotStyle"]

    plot_list = [kwargs["amplitudeTime"], kwargs["object"]]

    if numPlot == 1:
        axes = fig.subplots(ncols=1, nrows=1)
        axes[0].plot(
            xaxis[origin:endX],
            plot_list[0][origin:endX],
            plotStyle,
            label=plot_list[-1].name,
        )

    else:
        axes = fig.subplots(ncols=numPlot, nrows=1)
        for index_row, ax in enumerate(axes):
            ax.plot(
                xaxis[origin:endX],
                plot_list[0][index_row][origin:endX],
                plotStyle,
                label=plot_list[-1][index_row].name,
            )
            ax.grid()
            ax.legend(loc="lower center")

    plt.minorticks_on()
    plt.show()


def plot_time_domain_compare(xaxis, **kwargs):
    plot_list = [kwargs["amplitudeTime"], kwargs["object"]]

    lenX = kwargs["xlen"]
    if lenX == None:
        origin = None
        endX = None
    else:
        origin = int(len(xaxis) / 2)
        if lenX == -1:
            endX = -1
        else:
            endX = origin + lenX

    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    fig.set_facecolor("0.5")
    plotStyle = kwargs["plotStyle"]

    axes = fig.subplots(ncols=1, nrows=1)
    for index, yaxis in enumerate(plot_list[0]):
        axes.plot(
            xaxis[origin:endX],
            yaxis[origin:endX],
            plotStyle,
            label=plot_list[-1][index].name,
        )
    axes.legend(loc="upper right")
    axes.grid()

    # plt.legend(loc='upper right')
    plt.minorticks_on()
    plt.show()


def plot_transfer_response_compare(xaxis, **kwargs):
    plot_list = [kwargs["amplitude"], kwargs["phase"], kwargs["object"]]

    lenX = kwargs["xlen"]
    if lenX == None:
        origin = None
        endX = None
    else:
        origin = int(len(xaxis) / 2)
        if lenX == -1:
            endX = -1
        else:
            endX = origin + lenX

    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    fig.set_facecolor("0.5")
    plotStyle = kwargs["plotStyle"]

    axes = fig.subplots(ncols=1, nrows=2)
    for index, yaxis in enumerate(plot_list[0]):
        axes[0].plot(
            xaxis[origin:endX],
            yaxis[origin:endX],
            plotStyle,
            label=plot_list[-1][index].name,
        )
    axes[0].legend(loc="upper right")
    axes[0].grid()

    for index, yaxis in enumerate(plot_list[1]):
        axes[1].plot(
            xaxis[origin:endX],
            yaxis[origin:endX],
            plotStyle,
            label=plot_list[-1][index].name,
        )
    axes[1].legend(loc="upper right")
    axes[1].grid()

    # plt.legend(loc='upper right')
    plt.minorticks_on()
    plt.show()
