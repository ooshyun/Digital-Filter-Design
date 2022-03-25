"""Test for filter design
    This is testing file for filt.py, filter_analyze.py, filter_application.py,
    graphical_equalizer.py, fi.py which is on this projects. The entire object is
    Filter Design for Equalizer on the device.

    The types of tests is like below.
    - Frequency response for several types of filters
    - Floating to binary format using fi command
    - Cascade of filters's plot and processing
    - Parallel of filters's plot and processing
    
    ---
    TODO LIST
        [ ] 1. Frequency response for frames, wav file
"""
import os
import numpy as np
from numpy import log10, pi, sqrt

import matplotlib.pyplot as plt

import scipy.io.wavfile as wav
from scipy.fftpack import *

from lib.filter_analyze import FilterAnalyzePlot
from lib.filter_application import WaveProcessor
from lib.graphic_equalizer import GraphicalEqualizer
from lib.util import hilbert_from_scratch, cvt2float, cvt_pcm2wav, char2num
from lib.fi import fi

from lib.config import *

from lib.debug.log import PRINTER
from lib.debug.util import check_time, print_func_time


ROOT = os.getcwd()

def plot_cascade_sheving_filter():
    """ Cascade biquad 2nd order sheving filter
        Paper.
        SchultzHahnSpors_2020_ShelvingFiltersAdjustableTransitionBand_Paper.pdf
    """
    from scipy.signal import (
        bilinear_zpk,
        unit_impulse,
        sos2zpk,
        zpk2sos,
        sosfilt,
        sosfreqz,
    )
    from matplotlib.ticker import MultipleLocator
    from lib.util import (
        low_shelving_2nd_cascade,
        shelving_filter_parameters,
        db,
        set_rcparams,
        set_outdir,
        matchedz_zpk,
    )

    set_rcparams()

    fs = 48000
    fc = 1000
    w0 = 1

    # Frequency-domain evaluation
    wmin, wmax, num_w = 2 ** -9.5, 2 ** 4.5, 1000
    w = np.logspace(np.log10(wmin), np.log10(wmax), num=num_w)

    fmin, fmax, num_f = 10, 22000, 1000
    f = np.logspace(np.log10(fmin), np.log10(fmax), num=num_f)

    # Time-domain evaluation
    ws = 2 * np.pi * fs
    s2z = matchedz_zpk
    # s2z = bilinear_zpk

    fig, ax = plt.subplots(figsize=(13, 7), ncols=2, gridspec_kw={"wspace": 0.25})
    flim = fmin, fmax

    fticks = fc * 2.0 ** np.arange(-8, 4, 2)
    fticklabels = ["7.8", "31.3", "125", "500", "2k", "8k"]

    fticks = 1000 * 2.0 ** np.arange(-6, 5, 1)
    fticklabels = [
        "15.",
        "31.",
        "62.",
        "125",
        "250",
        "500",
        "1k",
        "2k",
        "4k",
        "8k",
        "16k",
    ]

    print(len(fticklabels), fticks)

    _Q = 1 / sqrt(2)

    biquad_per_octave = 1

    # _slope = -10 * np.log10(2)
    _slope = -10 * np.log10(2)
    _BWd = (
        None  # biquad_per_octave * Bwd / octave, expect 0db frequency fc/BWd = 250 Hz
    )
    _G_desire = 6

    n_list = np.arange(-3, 10, dtype=float)
    n_list = [pow(2, n) for n in n_list]
    n_list = list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    n_desire = []

    for n in n_list:

        print("-" * 10)
        print(n)
        print("-" * 10)
        Q = _Q
        BWd = _BWd
        G_desire = _G_desire
        slope = _slope * n

        # Analog filter
        H = np.zeros(num_w, dtype="complex")
        biquad_per_octave, num_biquad, Gb, G = shelving_filter_parameters(
            biquad_per_octave=biquad_per_octave, Gd=G_desire, slope=slope, BWd=BWd
        )

        if _G_desire is None:
            _G_desire = BWd * slope
        if slope is None:
            slope = G_desire / BWd

        sos_sdomain = low_shelving_2nd_cascade(
            w0, Gb, num_biquad, biquad_per_octave, Q=Q
        )
        zs, ps, ks = sos2zpk(sos_sdomain)

        # Digital filter s_domain * 2*pi*fc/fs
        zpk = s2z(zs * 2 * np.pi * fc, ps * 2 * np.pi * fc, ks, fs=fs)
        sos_zdomain = zpk2sos(*zpk)

        # print(sos_zdomain)

        w, H = sosfreqz(sos_zdomain, worN=f, fs=fs)

        # Plot
        def plot():
            # kw = dict(c='cornflowerblue', lw=2, alpha=1, label='S {:+0.2f}'.format(abs(slope)/3))
            kw = dict(c="cornflowerblue", lw=2, alpha=1)

            # frequency response

            # SLOPE (dB/dec) = ∆dB / (log10W2 – log10W1) = ∆dB / log10 (W2/W1);
            # ∆dB = Slope (dB/dec) * log10(W2/W1);
            # ∆dB = -20 (dB/dec) * log10(W2/W1)
            # idx_h0 = np.where(db(H) < 9)[0][0] # -3dB
            # # idx_h1 = idx_h0+1
            # idx_h1 = np.where(db(H) < 10e-2)[0][0]
            # f0 = f[idx_h0]
            # h0 = db(H)[idx_h0]
            # f1 = f[idx_h1]
            # h1 = db(H)[idx_h1]

            # print((h0, f0), (h1, f1))
            # print(abs(h0-h1)/log2(f1/f0))

            ax[0].semilogx(f, db(H), **kw)
            ax[1].semilogx(f, np.angle(H, deg=True), **kw)

        # plot()
        # if _G_desire > 0:
        #     if db(H).max() < _G_desire+3 and db(H)[0] > _G_desire-3:
        #         n_desire.append(log2(n))
        #         plot()

        # elif _G_desire < 0:
        #     if db(H).min() > _G_desire-3 and db(H)[0] < _G_desire+3:
        #         n_desire.append(log2(n))
        #         plot()

    # Gmin, Gmax = plt.ylim()

    from lib.biquad_cookbook import (
        shelf,
        peaking,
        bandpass,
        lowpass,
        notch,
        allpass,
        shelf,
    )
    from scipy.signal import freqz

    btype = "low"
    ftype = "half", "outer", "inner"
    slide = 1

    boost = _G_desire

    frame_size = 4096 * 8
    Wn = 2 * fc / fs  # 2 * pi * normalized_frequency

    fc = 1000 * 2 ** np.arange(-6, 5, dtype=float)
    Wn = 2 * fc / fs
    # b, a = peaking(Wn=wn, dBgain=boost, Q=1/sqrt(2), type='constantq',
    #                 analog=False, output='ba')
    # b, a = shelf(Wn=wn, dBgain=boost, S=1, btype=btype, ftype=ftype[0],
    #             analog=False, output='ba')
    # w, h = freqz(b, a, frame_size)
    # freq=w * fs * 1.0 / (2 * np.pi)

    for wn in Wn:
        print(wn)
        # b, a = shelf(Wn=wn, dBgain=g, Q=1/sqrt(2), btype=btype, ftype=ftype[0],
        #             analog=False, output='ba')
        b, a = lowpass(Wn=wn, Q=1 / sqrt(2), analog=False, output="ba")

        w, h = freqz(b, a, frame_size)
        freq = w * fs * 1.0 / (2 * np.pi)

        # ax[0].plot(freq, 20 * log10(abs(h)), label='Previous', color='orange')
        ax[0].semilogx(freq, 20 * log10(abs(h)))
        ax[1].semilogx(freq, np.angle(h, deg=True))

    ax[0].plot(freq, 20 * log10(abs(h)), color="orange", label="Previous")
    ax[1].plot(freq, np.angle(h, deg=True), color="orange", label="Previous")

    # decorations
    ax[0].set_xlim(flim)
    # plt.ylim(Gmin, Gmax)
    ax[0].set_xticks(fticks)
    ax[0].set_xticklabels(fticklabels)
    ax[0].minorticks_off()
    ax[0].grid(True)
    ax[0].set_xlabel("Frequency in Hz")
    ax[0].set_ylabel("Level in dB")

    ax[1].set_xlim(flim)
    # ax[1].set_ylim(phimin, phimax)
    ax[1].set_xticks(fticks)
    ax[1].set_xticklabels(fticklabels)
    ax[1].minorticks_off()
    # ax[1].yaxis.set_major_locator(MultipleLocator(15))
    # ax[1].yaxis.set_minor_locator(MultipleLocator(5))
    ax[1].grid(True)
    ax[1].set_xlabel("Frequency in Hz")
    ax[1].set_ylabel(r"Phase in degree")
    # ax[2].axis([0.76, 1.04, -0.14, 0.14])
    # ax[2].grid(True)
    # ax[2].set_aspect('equal')
    # ax[2].set_ylabel(r'$\Im (z)$')
    # ax[2].set_xlabel(r'$\Re (z)$')
    # ax[2].xaxis.set_major_locator(MultipleLocator(0.1))
    # ax[2].yaxis.set_major_locator(MultipleLocator(0.1))

    # ax[0].axvline(x=fc, color='r', label='cut-off frequency', linestyle='--', lw=2)
    # ax[0].axvline(x=fc, color='r', linestyle='--', lw=2)
    # ax[0].legend(loc='upper right', prop=dict(weight='bold'))
    # ax[1].axvline(x=fc, color='r', linestyle='--', lw=2)
    # ax[1].legend(loc='upper right', prop=dict(weight='bold'))

    plt.get_current_fig_manager().full_screen_toggle()

    plt.show()

    # outdir=''
    # name=f'shelf-filter-cascade'
    # plt.savefig(f'{outdir}/{name}.jpg')


def plot_sheving_filter_digital():
    """Filter design using biquad filter cookbook
        in digial domain
    """
    from lib.biquad_cookbook import shelf
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
        # ax[0].plot(freq, 20 * log10(abs(h)), 'b')
        # ax[1].plot(freq, angle, 'r')

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

    # plt.xlim(0.1, 1000)
    # ax[0].xlabel('Frequency [Hz]')
    # ax[0].ylabel('Amplitude [dB]')
    # ax[0].yticks(range(-24, 25, 3))
    # ax[0].margins(0, 0.1)
    # ax[0].grid(True, color='0.7', linestyle='-', which='major', axis='both')
    # ax[0].grid(True, color='0.9', linestyle='-', which='minor', axis='both')

    # plt.xlim(0.1, 1000)
    # ax[1].xlabel('Frequency [Hz]')
    # ax[1].ylabel('Amplitude [dB]')
    # ax[0].yticks(range(-24, 25, 3))
    # ax[1].margins(0, 0.1)
    # ax[1].grid(True, color='0.7', linestyle='-', which='major', axis='both')
    # ax[1].grid(True, color='0.9', linestyle='-', which='minor', axis='both')

    plt.show()

    # outdir=''
    # name=f'{test}_{btype}_gain_-24_24'
    # plt.savefig(f'{outdir}/{name}_S_{slide}.pdf')


def plot_sheving_filter_analog():
    """Filter design using biquad filter cookbook
        in analog domain
    """
    from lib.biquad_cookbook import peaking
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


def iir_filter_mulitstage_plot_cascade(fc_band):
    """Test frequency response for IIR filter cascade
    """
    from lib.filt import CustomFilter

    data_path = "/Users/seunghyunoh/workplace/study/FilterDesign/ExampleMusic/"
    result_path = (
        "/Users/seunghyunoh/workplace/study/filterdesign/resultmusic/serial_cascade/img"
    )

    infile_path = os.path.join(data_path, "White Noise.wav")
    fs, _ = wav.read(infile_path)

    ploter = FilterAnalyzePlot(sampleing_freq=fs)

    outfile_path = ""
    outresult_path = ""

    for f in fc_band:
        filter_custom = CustomFilter(
            sampling_freq=fs, cutoff_freq=f, Qfactor=4, gain=-6
        )

        peak_filter = filter_custom.peaking()
        ploter.filters = peak_filter

    ploter.series_equalizer = True

    path = os.path.join(result_path, "whitenoise_3peak_250_2000_8000.jpg")
    path = ""
    ploter.plot(type="gain", save_path=path)


def iir_filter_mulitstage_process_cascade(fc_band):
    """Test processing to wav for IIR filter cascade
    """
    from lib.filt import CustomFilter

    data_path = "/Users/seunghyunoh/workplace/study/FilterDesign/ExampleMusic/"
    result_path = "/Users/seunghyunoh/workplace/study/filterdesign/resultmusic/"

    infile_path = os.path.join(data_path, "test_noise_tx_in.wav")
    fs, _ = wav.read(infile_path)

    wave_processor = WaveProcessor(wavfile_path=infile_path)

    outfile_path = ""
    outresult_path = ""

    for f in fc_band:
        filter_custom = CustomFilter(sampling_freq=fs, cutoff_freq=f, Qfactor=4, gain=6)

        peak_filter = filter_custom.peaking()
        wave_processor.filters = peak_filter

    # wave_processor.graphical_equalizer = True
    wave_processor.run(
        savefile_path=result_path + "/whitenoise_3peak_250_2000_8000.wav"
    )

    # print(sum(wave_processor.freqfilter_time)/len(wave_processor.freqfilter_time))
    print(sum(wave_processor.timefilter_time) / len(wave_processor.timefilter_time))


def iir_filter_floating_format():
    """Test for iir filter in floating format"""

    white_noise = (
        "/Users/seunghyunoh/workplace/study/FilterDesign/ExampleMusic/music_sample.wav"
    )
    if not os.path.exists(white_noise.split(".")[0] + ".wav"):
        cvt_pcm2wav(white_noise, white_noise.split(".")[0] + ".wav", 48000, "int16")
    white_noise = white_noise.split(".")[0] + ".wav"

    print(f"processing {white_noise}......")

    wave_processor = WaveProcessor(wavfile_path=white_noise)
    ploter = FilterAnalyzePlot(sampleing_freq=wave_processor.sampleing_freq)

    # custom filter, Q 2.30, highpass
    alpha = [0x40000000, 0x84BCADBC, 0x3B6EA04E]
    beta = [0x3DAC7CA5, 0x84A706B8, 0x3DAC7CA5]

    a_local = np.array(
        fi(Values=alpha, Signed=1, TotalLen=32, FracLen=30, Format=0, ReturnVal="Dec")
    )
    b_local = np.array(
        fi(Values=beta, Signed=1, TotalLen=32, FracLen=30, Format=0, ReturnVal="Dec")
    )

    print(b_local, a_local)
    wave_processor.filters = b_local, a_local
    ploter.filters = b_local, a_local

    # Plot Frequency response
    ploter.plot(name="")

    # Process test
    # outfile_path = 'test.wav'
    # result_path = ''
    # outresult_path = os.path.join(result_path, outfile_path)
    # wave_processor.run(savefile_path=outresult_path)


def filter_coff_cvt2float(fc_band):
    from lib.filt import CustomFilter

    fs = 48000
    for fc in fc_band:
        filter_custom = CustomFilter(
            sampling_freq=fs, cutoff_freq=fc, Qfactor=4, gain=-6
        )
        filter = filter_custom.peaking()
        filter = np.array(filter).tolist()
        cvt2float(filter)


def lowpass_floating_cvt2binary32():
    coeff = [
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
    coeff = fi(coeff, 0, 32, 31, 0, "Dec")

    data_path = "/Users/seunghyunoh/workplace/study/FilterDesign/ExampleMusic/"
    result_path = "/Users/seunghyunoh/workplace/study/filterdesign/resultmusic/"

    infile_path = os.path.join(data_path, "test_noise_tx_in.wav")
    fs, data = wav.read(infile_path)

    wave_processor = WaveProcessor(wavfile_path=infile_path)
    wave_processor.sampleing_freq = fs
    for c in coeff:
        wave_processor.freqfilters = c
    wave_processor.run(savefile_path=result_path + "/test_noise_tx_in_lp_56.wav")


def iir_filter_mulitstage_plot_parallel(path, file):
    """Test the frequency response of parallel structure of iir filter 
    """
    # set the target
    data_path = "/Users/seunghyunoh/workplace/study/filter_design/ExampleMusic/"
    infile_path = os.path.join(data_path, "White Noise.wav")
    fs, data = wav.read(infile_path)

    # cuf-off freuqency case 1, single band
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

    # cut-off frequency case 2, double band
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

    for id_gain, gain in enumerate(gains):
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

        # set the gain type and one/double band type
        target_gain = gain_twice
        target_fc = fc_twice

        file_path = (path+file).split(".")[0]+"_"+str(id_gain+1)
        eq_set = GraphicalEqualizer(fs, target_fc, target_gain, Q)
        eq_set.freqz(save_path=file_path+".png")
        # eq_set.write_to_file(save_path=file_path+".txt")


def iir_filter_multistage_process_parallel(path, file):
    """Test wav file processing of parallel structure of iir filter 
    """
    data_path = "/Users/seunghyunoh/workplace/study/filter_design/ExampleMusic/"
    result_path = "/Users/seunghyunoh/workplace/study/filter_design/resultmusic/"

    src_wav = "White Noise.wav"
    infile_path = os.path.join(data_path, src_wav)

    PRINTER.info(f"source file {infile_path} is processing......")

    file_name = file.split(".")[0]
    for file_in_dir in os.listdir(path):
        if (file_name in file_in_dir) and (".txt" in file_in_dir):
            _file = os.path.join(path, file_in_dir)
            PRINTER.info(f"{_file} is processing......")

            coeff_text = open(_file).read()
            coeff_text = coeff_text.split("\n")[:-1]
            coeff_text = [text.split(" ") for text in coeff_text]
            char2num(coeff_text)

            coeff_text, bias = np.array(coeff_text[:-1]), np.array(coeff_text[-1])

            wave_processor = WaveProcessor(wavfile_path=infile_path)
            wave_processor.graphical_equalizer = True
            wave_processor.filters = coeff_text
            wave_processor.bias = bias

            outresult_path = (
                result_path + "/whitenoise_" + _file.split(".")[-2].split("/")[-1] + ".wav"
            )

            PRINTER.info(f"target file {outresult_path} is processing......")

            wave_processor.run(savefile_path=outresult_path)
        else:
            PRINTER.info(f"{file_in_dir} was passed")
            continue


if __name__ == "__main__":
    PRINTER.info("Hello Digital Signal Processing World!")

    fc_band = np.array([250, 2000, 8000])
    """ Several types of filters and
        Analysis analog and digital scope of pole and zeros """
    # plot_cascade_sheving_filter()
    # plot_sheving_filter_digital()
    # plot_sheving_filter_analog()
    """Cascade of filters coffcient design"""
    # iir_filter_mulitstage_plot_cascade(fc_band)
    # iir_filter_mulitstage_process_cascade(fc_band)
    """ RTK floating <-> binary floating format parameter test """
    # iir_filter_floating_format()
    # filter_coff_cvt2float(fc_band)
    # lowpass_floating_cvt2binary32()
    """Parallel of filters coffcient design"""
    file_path = os.path.join(ROOT, "lib/data/")
    file_name = "coeff_test.txt"
    iir_filter_mulitstage_plot_parallel(file_path, file_name)
    # iir_filter_multistage_process_parallel(file_path, file_name)
