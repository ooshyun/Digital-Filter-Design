import matplotlib.pyplot as plt
from scipy.fftpack import *
import scipy.io.wavfile as wav
import numpy as np
from numpy import log10, pi, sqrt
import matplotlib.pyplot as plt


from src import (
    FilterAnalyzePlot,
    WaveProcessor,
    cvt_float2fixed,
    fi,
)

LIBRARY_PATH = "./" # First of all, it need to set the library(or this project) path


def example_fi():
    """Test fi command
    """
    """ first example"""
    x = 0xF9E68000  # Q 1.31
    b = 0x3DAC7CA5  # Q 2.30
    mul_x_b = 0xFFFD0FA0D34C00  # Q 2.46

    # output_values=fi(input_values,1,32,31,0,ReturnVal='Dec')
    x = fi(x, 1, 32, 31, 0, "Dec")[0]
    b = fi(b, 1, 32, 30, 0, "Dec")[0]
    mul_x_b = fi(mul_x_b, 1, 56, 46, 0, "Dec")[0]

    print("-" * 10)
    if x * b - mul_x_b < 10e-3:
        print(f"PASS, {mul_x_b}")

    x = 0xF9E68000  # Q 1.31
    b = 0x3DAC7CA5  # Q 2.30
    mul_x_b = 0xFFFD0FA0D34C00  # Q 2.46
    y_shift = 0xFFFA1F41A69800  # Q 1.47

    y = 0xFA1F4100  # Q 1.31
    """ second example """
    x = 0xF9E68000  # Q 1.31
    b = 0x84A70600  # Q 2.30

    mul_x_b = 0x0005E0BE71CE00  # Q 2.46

    x = fi(x, 1, 32, 31, 0, "Dec")[0]
    b = fi(b, 1, 32, 30, 0, "Dec")[0]
    mul_x_b = fi(mul_x_b, 1, 56, 46, 0, "Dec")[0]

    print("-" * 10)
    if x * b - mul_x_b < 10e-3:
        print(f"PASS, {mul_x_b}")

    y = 0xFA1F4100  # Q 1.31
    a = 0x84BCAD00  # Q 2.30
    mul_y_a = 0x0005A90EFDB5DA  # Q 2.46

    y = fi(y, 1, 32, 31, 0, "Dec")[0]
    a = fi(a, 1, 32, 30, 0, "Dec")[0]
    mul_y_a = fi(mul_y_a, 1, 56, 46, 0, "Dec")[0]

    print("-" * 10)
    if y * y - mul_y_a < 10e-3:
        print(f"PASS, {mul_y_a}")

    zi_predict = mul_x_b - mul_y_a
    mul_x_b = 0x0005E0BE71CE00  # Q 2.46
    mul_y_a = 0x0005A90EFDB5DA  # Q 2.46
    zi = mul_x_b - mul_y_a
    zi = fi(zi, 1, 56, 46, 0, "Dec")[0]
    print("-" * 10)
    if zi_predict - zi < 10e-3:
        print(f"PASS, {zi}")

    """final example"""
    x = 0xF9E68000  # Q 1.31
    b = 0x3DAC7C00  # Q 2.30

    mul_x_b = 0xFFFD0FA0D34C00  # Q 2.46

    x = fi(x, 1, 32, 31, 0, "Dec")[0]
    b = fi(b, 1, 32, 30, 0, "Dec")[0]
    mul_x_b = fi(mul_x_b, 1, 56, 46, 0, "Dec")[0]

    print("-" * 10)
    if x * b - mul_x_b < 10e-3:
        print(f"PASS, {mul_x_b}")

    y = 0xFA1F4100  # Q 1.31
    a = 0x3B6EA000  # Q 2.30
    mul_y_a = 0xFFFD455378ED40  # Q 2.46

    y = fi(y, 1, 32, 31, 0, "Dec")[0]
    a = fi(a, 1, 32, 30, 0, "Dec")[0]
    mul_y_a = fi(mul_y_a, 1, 56, 46, 0, "Dec")[0]

    print("-" * 10)
    if y * a - mul_y_a < 10e-3:
        print(f"PASS, {mul_y_a}")
    else:
        print(f"FAIL, {mul_y_a}")

    zi_predict = mul_x_b - mul_y_a
    mul_x_b = 0xFFFD0FA0D34C00  # Q 2.46
    mul_y_a = 0xFFFD455378ED40  # Q 2.46

    zi = mul_x_b - mul_y_a
    zi = fi(zi, 1, 56, 46, 0, "Dec")[0]
    _zi = 0xFFFFCA4D5A5EC0
    print("-" * 10)
    if zi_predict - zi < 10e-3:
        print(f"PASS, {zi_predict, zi}")
    else:
        print(f"FAIL, {zi_predict, zi}")

    # 0, 1, 2, 3, 4,
    gainDb = [-5912, 0]  # Q7 , int16_t
    gain = [159, 0]  # Q15, int16_t
    print(fi(gain, 1, 16, 8, 0, "Dec"))

    gain = 0x7FFF

    print(fi(gain, 1, 16, 15, 0, "Dec"))

    # convert text hex of fixed point to floating point, Q 1.31, data is hanning window
    data = "0x00000000,0x0004f92c,0x0013e3eb,0x002cbdeb,0x004f834f,0x007c2eb0,0x00b2b91c,0x00f31a1b,0x013d47a9,0x01913640,0x01eed8d5,0x025620da,0x02c6fe42,0x03415f82,0x03c53195,0x04526000,0x04e8d4cf,0x058878a2,0x063132a9,0x06e2e8aa,0x079d7f08,0x0860d8c2,0x092cd77c,0x0a015b82,0x0ade43cc,0x0bc36e07,0x0cb0b692,0x0da5f88f,0x0ea30ddf,0x0fa7cf2d,0x10b413f1,0x11c7b27c,0x12e27ff5,0x1404506b,0x152cf6d2,0x165c4510,0x17920c00,0x18ce1b7e,0x1a10426b,0x1b584eb6,0x1ca60d62,0x1df94a92,0x1f51d18b,0x20af6cc3,0x2211e5e3,0x237905d5,0x24e494c7,0x26545a3a,0x27c81d05,0x293fa360,0x2abab2ef,0x2c3910c9,0x2dba817f,0x2f3ec92c,0x30c5ab76,0x324eeb9d,0x33da4c84,0x356790b7,0x36f67a79,0x3886cbca,0x3a184673,0x3baaac0f,0x3d3dbe13,0x3ed13ddb,0x4064ecb0,0x41f88bd6,0x438bdc92,0x451ea035,0x46b09827,0x484185ee,0x49d12b3b,0x4b5f49f2,0x4ceba432,0x4e75fc63,0x4ffe1539,0x5183b1c6,0x5306957c,0x54868439,0x56034253,0x577c949d,0x58f24073,0x5a640bc0,0x5bd1bd0e,0x5d3b1b85,0x5e9feefd,0x60000000,0x615b17d7,0x62b10090,0x64018507,0x654c70f1,0x669190de,0x67d0b247,0x6909a392,0x6a3c341e,0x6b683444,0x6c8d7565,0x6dabc9ed,0x6ec3055c,0x6fd2fc4c,0x70db8479,0x71dc74c5,0x72d5a543,0x73c6ef37,0x74b02d22,0x75913ac4,0x7669f522,0x773a3a8e,0x7801eaa9,0x78c0e66b,0x79771024,0x7a244b86,0x7ac87da3,0x7b638cf8,0x7bf5616a,0x7c7de450,0x7cfd0072,0x7d72a20f,0x7ddeb6df,0x7e412e15,0x7e99f865,0x7ee90801,0x7f2e50a0,0x7f69c77d,0x7f9b635a,0x7fc31c83,0x7fe0ecc9,0x7ff4cf8b,0x7ffec1b2,0x7ffec1b2,0x7ff4cf8b,0x7fe0ecc9,0x7fc31c83,0x7f9b635a,0x7f69c77d,0x7f2e50a0,0x7ee90801,0x7e99f865,0x7e412e15,0x7ddeb6df,0x7d72a20f,0x7cfd0072,0x7c7de450,0x7bf5616a,0x7b638cf8,0x7ac87da3,0x7a244b86,0x79771024,0x78c0e66b,0x7801eaa9,0x773a3a8e,0x7669f522,0x75913ac4,0x74b02d22,0x73c6ef37,0x72d5a543,0x71dc74c5,0x70db8479,0x6fd2fc4c,0x6ec3055c,0x6dabc9ed,0x6c8d7565,0x6b683444,0x6a3c341e,0x6909a392,0x67d0b247,0x669190de,0x654c70f1,0x64018507,0x62b10090,0x615b17d7,0x60000000,0x5e9feefd,0x5d3b1b85,0x5bd1bd0e,0x5a640bc0,0x58f24073,0x577c949d,0x56034253,0x54868439,0x5306957c,0x5183b1c6,0x4ffe1539,0x4e75fc63,0x4ceba432,0x4b5f49f2,0x49d12b3b,0x484185ee,0x46b09827,0x451ea035,0x438bdc92,0x41f88bd6,0x4064ecb0,0x3ed13ddb,0x3d3dbe13,0x3baaac0f,0x3a184673,0x3886cbca,0x36f67a79,0x356790b7,0x33da4c84,0x324eeb9d,0x30c5ab76,0x2f3ec92c,0x2dba817f,0x2c3910c9,0x2abab2ef,0x293fa360,0x27c81d05,0x26545a3a,0x24e494c7,0x237905d5,0x2211e5e3,0x20af6cc3,0x1f51d18b,0x1df94a92,0x1ca60d62,0x1b584eb6,0x1a10426b,0x18ce1b7e,0x17920c00,0x165c4510,0x152cf6d2,0x1404506b,0x12e27ff5,0x11c7b27c,0x10b413f1,0x0fa7cf2d,0x0ea30ddf,0x0da5f88f,0x0cb0b692,0x0bc36e07,0x0ade43cc,0x0a015b82,0x092cd77c,0x0860d8c2,0x079d7f08,0x06e2e8aa,0x063132a9,0x058878a2,0x04e8d4cf,0x04526000,0x03c53195,0x03415f82,0x02c6fe42,0x025620da,0x01eed8d5,0x01913640,0x013d47a9,0x00f31a1b,0x00b2b91c,0x007c2eb0,0x004f834f,0x002cbdeb,0x0013e3eb,0x0004f92c,0x00000000"
    data = data.split(",")
    hann256_fixed = [int(num, 0) for num in data]  # consider "0x"
    hann256_floating = fi(
        hann256_fixed, Signed=True, TotalLen=32, FracLen=31, Format=0, ReturnVal="Dec"
    )

    # convert floating point to fixed point, Q 1.31, data is hanning window
    hann256_floating = 0.5 * (1 - np.cos((2 * np.pi * np.arange(0, 256)) / 255))
    hann256_fixed = fi(
        hann256_floating.tolist(),
        Signed=False,
        TotalLen=32,
        FracLen=31,
        Format=1,
        ReturnVal="Hex",
    ).split(",")


def iir_filter_fixed_to_floating_format_plot():
    """Test for iir filter in floating format"""
    ploter = FilterAnalyzePlot(sample_rate=48000)

    # custom filter, Q 2.30, highpass
    alpha = [0x40000000, 0x84BCADBC, 0x3B6EA04E]
    beta = [0x3DAC7CA5, 0x84A706B8, 0x3DAC7CA5]

    a_local = np.array(
        fi(Values=alpha, Signed=1, TotalLen=32, FracLen=30, Format=0, ReturnVal="Dec")
    )
    b_local = np.array(
        fi(Values=beta, Signed=1, TotalLen=32, FracLen=30, Format=0, ReturnVal="Dec")
    )

    ploter.filters = b_local, a_local

    # Plot Frequency response
    ploter.plot(type=["freq", "phase", "pole"])


def iir_filter_floating_to_fixed_format_print():
    from src import peaking

    fs = 48000
    fc_band = np.array([250, 2000, 8000])
    for fc in fc_band:
        filter = peaking(Wn=2 * fc / fs, dBgain=-6, Q=4)
        filter = np.array(filter).tolist()
        cvt_float2fixed(filter)


def iir_filter_fixed_to_floating_format_process():
    coeff_freq_domain = [
        0x5A9DF7AC,
        0x4026E73D,
        0x2D6A8670,
        0x2026F310,
        0x16C310E4,
        0x101D3F2E,
        0x0B68737A,
        0x08138562,
        0x05B7B15B,
        0x040C3714,
        0x02DD958B,
        0x0207567B,
        0x016FA9BB,
        0x01044915,
        0x00B8449D,
        0x008273A7,
        0x005C5A50,
        0x0041617A,
        0x002E493A,
        0x0020C49C,
        0x001732AE,
        0x00106C44,
        0x000BA064,
        0x00083B20,
        0x0005D3BB,
        0x00042011,
        0x0002EBA3,
        0x0002114A,
        0x000176B5,
        0x00010946,
        0x0000BBCD,
        0x000084F4,
        0x00005E20,
        0x000042A3,
        0x00002F2D,
        0x00002166,
        0x000017A5,
        0x000010BD,
        0x00000BDA,
        0x00000864,
        0x000005F1,
        0x00000435,
        0x000002FA,
        0x0000021C,
        0x0000017E,
        0x0000010F,
        0x000000C0,
        0x00000088,
        0x00000060,
        0x00000044,
        0x00000031,
        0x00000023,
        0x00000019,
        0x00000012,
        0x0000000D,
        0x00000009,
        0x00000007,
        0x00000005,
        0x00000004,
        0x00000003,
        0x00000002,
        0x00000002,
        0x00000001,
        0x00000000,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    # Q 1.31, Lowpass
    coeff_freq_domain = fi(coeff_freq_domain, 0, 32, 31, 0, "Dec")
    coeff_freq_domain = np.array(coeff_freq_domain)
    infile_path = LIBRARY_PATH + "/test/data/wav/White Noise.wav"
    outfile_path = (
        LIBRARY_PATH + "/test/result/wav/White Noise_lowpass_floating_cvt2binary32.wav"
    )
    fs, data = wav.read(infile_path)

    nfft = 256  # It is already designed
    coeff_frequency = np.zeros(shape=(nfft // 2 + 1,))
    coeff_frequency[: len(coeff_freq_domain)] = coeff_freq_domain

    wave_processor = WaveProcessor(wavfile_path=infile_path)
    wave_processor.sampleing_freq = fs
    wave_processor.filter_freq_domain_list = coeff_frequency
    wave_processor.run(savefile_path=outfile_path)


def scale_logarithmic():
    # frequency = np.arange(20, 22051, 1)
    f0 = 20
    fs = 44100
    f0_log = np.log10(f0)
    frequency_log = np.arange(f0_log, np.log10(fs / 2), 0.1)
    frequency = np.power(10, frequency_log)
    # frequency = 20*10**(np.arange(0, 10, 0.1))
    ideal_frequency = np.array(
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
    fig = plt.figure(figsize=(8, 6))

    axes = [0] * 3
    axes[0] = fig.add_subplot(2, 1, 1)
    axes[1] = fig.add_subplot(2, 2, 3)
    axes[2] = fig.add_subplot(2, 2, 4)

    axes[0].plot(frequency, ".")
    axes[0].plot(ideal_frequency, "*")

    axes[1].plot(frequency, ".")
    axes[2].plot(ideal_frequency, "*")

    plt.show()


def plot_sheving_filter_digital():
    """Filter design using biquad filter cookbook
        in digial domain
    """
    from src import shelf
    from scipy.signal import freqz

    """ Basic 2nd order sheving Filter """
    filter_type = {"low_pass", "high_pass", "all_pass", "bandpass", "peaking", "shelf"}

    test = "shelf"

    btype = "low"
    ftype = "half"
    slide = 1

    fc = 1000
    fs = 44100
    frame_size = 1024

    wn = fc / fs / 2

    fig = plt.figure(figsize=(14, 7))
    # ax = fig.subplots(ncols=1, nrows=2)

    for boost in range(-24, 0, 3):
        b, a = shelf(
            Wn=wn,
            dBgain=boost,
            S=slide,
            btype=btype,
            ftype=ftype,
            analog=False,
            output="ba",
        )
        w, h = freqz(b, a, frame_size)
        freq = w * fs * 1.0 / (2 * np.pi)
        angle = 180 * np.angle(h) / pi  # Convert to degrees

        plt.plot(freq, 20 * log10(abs(h)), "r")

    for boost in range(0, 25, 3):
        b, a = shelf(
            Wn=wn,
            dBgain=boost,
            S=slide,
            btype=btype,
            ftype=ftype,
            analog=False,
            output="ba",
        )
        w, h = freqz(b, a, frame_size)
        freq = w * fs * 1.0 / (2 * np.pi)
        angle = 180 * np.angle(h) / pi  # Convert to degrees

        plt.plot(freq, 20 * log10(abs(h)), "b")

    plt.xscale("log")
    plt.title(f"shelving filter, Frequency Response")
    plt.xlim(0.1, 1000)
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


def plot_sheving_filter_analog():
    """Filter design using biquad filter cookbook
        in analog domain
    """
    from src import peaking
    from scipy.signal import freqs

    for ftype in ("half", "constantq"):
        plt.figure()
        for boost in range(-24, 0, 3):
            b, a = peaking(10, dBgain=boost, Q=sqrt(2), type=ftype, analog=True)
            w, h = freqs(b, a, 10000)
            plt.plot(w, 20 * log10(abs(h)), "r", alpha=0.5)

        for boost in range(0, 25, 3):
            b, a = peaking(10, dBgain=boost, Q=sqrt(2), type=ftype, analog=True)
            w, h = freqs(b, a, 10000)
            plt.plot(w, 20 * log10(abs(h)), "b", alpha=0.5)

        plt.xscale("log")
        plt.title(f'Peaking filter, "{ftype}" frequency response')
        plt.xlim(0.1, 1000)
        plt.xlabel("Frequency [radians / second]")
        plt.ylabel("Amplitude [dB]")
        plt.yticks(range(-24, 25, 3))
        plt.margins(0, 0.1)
        plt.grid(True, color="0.7", linestyle="-", which="major", axis="both")
        plt.grid(True, color="0.9", linestyle="-", which="minor", axis="both")
        plt.show()


def paper_all_pass_filter():
    """paper. frequency-warped signal preprocessing for audio application
        - 7p. 2.1 Allpass filter chain
        
        fig.add_subplot(total rows, total cols, index)
        index = n_row * n_col + n_col
    """
    import scipy.signal as signal

    sampling_rate = 44100
    dt = 1 / sampling_rate
    w = np.linspace(0, np.pi, sampling_rate // 2, endpoint=True)
    f = np.linspace(0, sampling_rate // 2, sampling_rate // 2, endpoint=True)
    z = np.exp(-1j * w)  # z^-1
    _lambda = 0

    fig = plt.figure(figsize=(8, 6))

    axes = [0] * 3
    axes[0] = fig.add_subplot(2, 2, 3)
    axes[1] = fig.add_subplot(2, 2, 4)
    axes[2] = fig.add_subplot(2, 1, 1)

    for _lambda in [0, 0.5, 0.723, 0.9]:
        b = [-_lambda, 1]
        a = [1, -_lambda]
        D_z = (-_lambda + z) / (1 - _lambda * z)

        # phase
        phi = np.angle(D_z) / (2 * np.pi)
        # phase normalize
        phi /= phi[np.where(np.abs(phi) == np.max(np.abs(phi)))[0][0]]
        axes[0].plot(f, phi, ".", label=f"lambda={_lambda}")

        # group delay
        w, gd = signal.group_delay((b, a), w=sampling_rate // 2)
        axes[1].plot(f, gd, "*", label=f"lambda={_lambda}")

        # amplitude
        amplitude = np.abs(D_z)
        axes[2].plot(f, amplitude, ".", label=f"lambda={_lambda}")

    axes[0].set_ylim(0, 1)
    axes[0].set_xlim(0, sampling_rate // 2)
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title("phase normalize")

    axes[1].set_ylim(0, 7)
    axes[1].set_xlim(0, sampling_rate // 2)
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title("group delay")

    axes[2].set_yticks(np.arange(0, 2.5, 0.5))
    axes[2].set_xlim(0, sampling_rate // 2)
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_title("amplitude")

    # plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    """ 1. Fi command example: fixed point <-> floating format parameter test """
    example_fi()
    iir_filter_fixed_to_floating_format_plot()
    iir_filter_floating_to_fixed_format_print()
    iir_filter_fixed_to_floating_format_process()

    """ 2. Logarithmic scale"""
    scale_logarithmic()

    """ 3. Analying the difference between digital and analog """
    plot_sheving_filter_digital()
    plot_sheving_filter_analog()

    """4. Frequency wrapping """
    paper_all_pass_filter()
    pass
