"""Test for filter design
    This is testing file for filt.py, filter_analyze.py, filter_application.py,
    graphical_equalizer.py, fi.py which is on this projects. The entire object is
    Filter Design for Equalizer on the device.

    The types of tests is like below.
    - Frequency response for several types of filters
    - Fi command
    - Floating to binary format using fi command
    - Cascade of filters's plot and processing
    - 2nd interpolation
    - Hibert transform from signal in time-domain
        and amplitude frequency-domain
    - Make Matrix to Even-odd cross matrix
    - Parallel of filters's plot and processing
    - Frequency response from Audacity frequency response
    
    ---
    TODO LIST
        [ ] 1. Frequency response for frames, wav file
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.fftpack import *
from numpy import dtype, log10, pi, log2, sqrt

from lib.filter_analyze import FilterAnalyzePlot
from lib.filter_application import WaveProcessor
from lib.graphic_equalizer import GraphicalEqualizer
from lib.debug.log import maker_logger
import scipy.signal
import math, time

from lib.util import hilbert_from_scratch, cvt2float, cvt_pcm2wav, char2num
from lib.fi import fi

ROOT = os.getcwd()


def plot_cascade_sheving_filter():
    """ Cascade biquad 2nd order sheving filter
        Paper.
        SchultzHahnSpors_2020_ShelvingFiltersAdjustableTransitionBand_Paper.pdf
    """
    from scipy.signal import bilinear_zpk, unit_impulse, sos2zpk, zpk2sos,\
                            sosfilt, sosfreqz
    from matplotlib.ticker import MultipleLocator
    from lib.util import low_shelving_2nd_cascade, shelving_filter_parameters,\
                    db, set_rcparams, set_outdir, matchedz_zpk

    set_rcparams()

    fs = 48000
    fc = 1000
    w0 = 1

    # Frequency-domain evaluation
    wmin, wmax, num_w = 2**-9.5, 2**4.5, 1000
    w = np.logspace(np.log10(wmin), np.log10(wmax), num=num_w)

    fmin, fmax, num_f = 10, 22000, 1000
    f = np.logspace(np.log10(fmin), np.log10(fmax), num=num_f)

    # Time-domain evaluation
    ws = 2 * np.pi * fs
    s2z = matchedz_zpk
    # s2z = bilinear_zpk

    fig, ax = plt.subplots(figsize=(13, 7),
                           ncols=2,
                           gridspec_kw={'wspace': 0.25})
    flim = fmin, fmax

    fticks = fc * 2.**np.arange(-8, 4, 2)
    fticklabels = ['7.8', '31.3', '125', '500', '2k', '8k']

    fticks = 1000 * 2.**np.arange(-6, 5, 1)
    fticklabels = [
        '15.', '31.', '62.', '125', '250', '500', '1k', '2k', '4k', '8k', '16k'
    ]

    print(len(fticklabels), fticks)

    _Q = 1 / sqrt(2)

    biquad_per_octave = 1

    # _slope = -10 * np.log10(2)
    _slope = -10 * np.log10(2)
    _BWd = None    # biquad_per_octave * Bwd / octave, expect 0db frequency fc/BWd = 250 Hz
    _G_desire = 6

    n_list = np.arange(-3, 10, dtype=float)
    n_list = [pow(2, n) for n in n_list]
    n_list = list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    n_desire = []

    for n in n_list:

        print('-' * 10)
        print(n)
        print('-' * 10)
        Q = _Q
        BWd = _BWd
        G_desire = _G_desire
        slope = _slope * n

        # Analog filter
        H = np.zeros(num_w, dtype='complex')
        biquad_per_octave, num_biquad, Gb, G = shelving_filter_parameters(
            biquad_per_octave=biquad_per_octave,
            Gd=G_desire,
            slope=slope,
            BWd=BWd)

        if _G_desire is None:
            _G_desire = BWd * slope
        if slope is None:
            slope = G_desire / BWd

        sos_sdomain = low_shelving_2nd_cascade(w0,
                                               Gb,
                                               num_biquad,
                                               biquad_per_octave,
                                               Q=Q)
        zs, ps, ks = sos2zpk(sos_sdomain)

        # Digital filter s_domain * 2*pi*fc/fs
        zpk = s2z(zs * 2 * np.pi * fc, ps * 2 * np.pi * fc, ks, fs=fs)
        sos_zdomain = zpk2sos(*zpk)

        # print(sos_zdomain)

        w, H = sosfreqz(sos_zdomain, worN=f, fs=fs)

        # Plot
        def plot():
            # kw = dict(c='cornflowerblue', lw=2, alpha=1, label='S {:+0.2f}'.format(abs(slope)/3))
            kw = dict(c='cornflowerblue', lw=2, alpha=1)

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

    from lib.biquad_cookbook import shelf, peaking, bandpass, lowpass, \
                                notch, allpass, shelf
    from scipy.signal import freqz

    btype = 'low'
    ftype = 'half', 'outer', 'inner'
    slide = 1

    boost = _G_desire

    frame_size = 4096 * 8
    Wn = 2 * fc / fs    # 2 * pi * normalized_frequency

    fc = 1000 * 2**np.arange(-6, 5, dtype=float)
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
        b, a = lowpass(Wn=wn, Q=1 / sqrt(2), analog=False, output='ba')

        w, h = freqz(b, a, frame_size)
        freq = w * fs * 1.0 / (2 * np.pi)

        # ax[0].plot(freq, 20 * log10(abs(h)), label='Previous', color='orange')
        ax[0].semilogx(freq, 20 * log10(abs(h)))
        ax[1].semilogx(freq, np.angle(h, deg=True))

    ax[0].plot(freq, 20 * log10(abs(h)), color='orange', label='Previous')
    ax[1].plot(freq, np.angle(h, deg=True), color='orange', label='Previous')

    # decorations
    ax[0].set_xlim(flim)
    # plt.ylim(Gmin, Gmax)
    ax[0].set_xticks(fticks)
    ax[0].set_xticklabels(fticklabels)
    ax[0].minorticks_off()
    ax[0].grid(True)
    ax[0].set_xlabel('Frequency in Hz')
    ax[0].set_ylabel('Level in dB')

    ax[1].set_xlim(flim)
    # ax[1].set_ylim(phimin, phimax)
    ax[1].set_xticks(fticks)
    ax[1].set_xticklabels(fticklabels)
    ax[1].minorticks_off()
    # ax[1].yaxis.set_major_locator(MultipleLocator(15))
    # ax[1].yaxis.set_minor_locator(MultipleLocator(5))
    ax[1].grid(True)
    ax[1].set_xlabel('Frequency in Hz')
    ax[1].set_ylabel(r'Phase in degree')
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
    filter_type = {
        'low_pass', 'high_pass', 'all_pass', 'bandpass', 'peaking', 'shelf'
    }

    test = 'shelf'

    btype = 'low'
    ftype = 'half'
    slide = 1

    fc = 1000
    fs = 44100
    frame_size = 1024

    wn = fc / fs / 2

    fig = plt.figure(figsize=(14, 7))
    # ax = fig.subplots(ncols=1, nrows=2)

    for boost in range(-24, 0, 3):
        b, a = shelf(Wn=wn,
                     dBgain=boost,
                     S=slide,
                     btype=btype,
                     ftype=ftype,
                     analog=False,
                     output='ba')
        w, h = freqz(b, a, frame_size)
        freq = w * fs * 1.0 / (2 * np.pi)

        angle = 180 * np.angle(h) / pi    # Convert to degrees

        plt.plot(freq, 20 * log10(abs(h)), 'r')

    for boost in range(0, 25, 3):
        b, a = shelf(Wn=wn,
                     dBgain=boost,
                     S=slide,
                     btype=btype,
                     ftype=ftype,
                     analog=False,
                     output='ba')
        w, h = freqz(b, a, frame_size)
        freq = w * fs * 1.0 / (2 * np.pi)

        angle = 180 * np.angle(h) / pi    # Convert to degrees

        plt.plot(freq, 20 * log10(abs(h)), 'b')
        # ax[0].plot(freq, 20 * log10(abs(h)), 'b')
        # ax[1].plot(freq, angle, 'r')

    plt.xscale('log')
    plt.title(f'shelving filter, Frequency Response')
    plt.xlim(0.1, 1000)
    plt.xticks(
        [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
        ["10", "50", "100", "200", "500", "1K", "2K", "5K", "10K", "20K"])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.yticks(range(-24, 25, 3))
    plt.margins(0, 0.1)
    plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
    plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')

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

    for ftype in ('half', 'constantq'):
        plt.figure()
        for boost in range(-24, 0, 3):
            b, a = peaking(10, dBgain=boost, Q=sqrt(2), type=ftype, analog=True)
            w, h = freqs(b, a, 10000)
            plt.plot(w, 20 * log10(abs(h)), 'r', alpha=0.5)

        for boost in range(0, 25, 3):
            b, a = peaking(10, dBgain=boost, Q=sqrt(2), type=ftype, analog=True)
            w, h = freqs(b, a, 10000)
            plt.plot(w, 20 * log10(abs(h)), 'b', alpha=0.5)

        plt.xscale('log')
        plt.title(f'Peaking filter, "{ftype}" frequency response')
        plt.xlim(0.1, 1000)
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        plt.yticks(range(-24, 25, 3))
        plt.margins(0, 0.1)
        plt.grid(True, color='0.7', linestyle='-', which='major', axis='both')
        plt.grid(True, color='0.9', linestyle='-', which='minor', axis='both')
        plt.show()


def iir_filter_mulitstage_plot_cascade(fc_band):
    """Test frequency response for IIR filter cascade
    """
    from lib.filt import CustomFilter
    data_path = '/Users/seunghyunoh/workplace/study/FilterDesign/ExampleMusic/'
    result_path = '/Users/seunghyunoh/workplace/study/filterdesign/resultmusic/serial_cascade/img'

    infile_path = os.path.join(data_path, 'White Noise.wav')
    fs, _ = wav.read(infile_path)

    ploter = FilterAnalyzePlot(sampleing_freq=fs)

    outfile_path = ''
    outresult_path = ''

    for f in fc_band:
        filter_custom = CustomFilter(sampling_freq=fs,
                                     cutoff_freq=f,
                                     Qfactor=4,
                                     gain=-6)

        peak_filter = filter_custom.peaking()
        ploter.filters = peak_filter

    ploter.series_equalizer = True

    path = os.path.join(result_path, 'whitenoise_3peak_250_2000_8000.jpg')
    path = ''
    ploter.plot(type='gain', save_path=path)


def iir_filter_mulitstage_process_cascade(fc_band):
    """Test processing to wav for IIR filter cascade
    """
    from lib.filt import CustomFilter
    data_path = '/Users/seunghyunoh/workplace/study/FilterDesign/ExampleMusic/'
    result_path = '/Users/seunghyunoh/workplace/study/filterdesign/resultmusic/'

    infile_path = os.path.join(data_path, 'test_noise_tx_in.wav')
    fs, _ = wav.read(infile_path)

    wave_processor = WaveProcessor(wavfile_path=infile_path)

    outfile_path = ''
    outresult_path = ''

    for f in fc_band:
        filter_custom = CustomFilter(sampling_freq=fs,
                                     cutoff_freq=f,
                                     Qfactor=4,
                                     gain=6)

        peak_filter = filter_custom.peaking()
        wave_processor.filters = peak_filter

    # wave_processor.graphical_equalizer = True
    wave_processor.run(savefile_path=result_path +
                       '/whitenoise_3peak_250_2000_8000.wav')

    # print(sum(wave_processor.freqfilter_time)/len(wave_processor.freqfilter_time))
    print(
        sum(wave_processor.timefilter_time) /
        len(wave_processor.timefilter_time))


def fi_example():
    """Test fi command
    """
    ''' first '''
    x = 0xf9e68000    # Q 1.31
    b = 0x3dac7ca5    # Q 2.30
    mul_x_b = 0xfffd0fa0d34c00    # Q 2.46

    # output_values=fi(input_values,1,32,31,0,ReturnVal='Dec')
    x = fi(x, 1, 32, 31, 0, 'Dec')[0]
    b = fi(b, 1, 32, 30, 0, 'Dec')[0]
    mul_x_b = fi(mul_x_b, 1, 56, 46, 0, 'Dec')[0]

    print("-" * 10)
    if (x * b - mul_x_b < 10e-3):
        print(f'PASS, {mul_x_b}')

    x = 0xf9e68000    # Q 1.31
    b = 0x3dac7ca5    # Q 2.30
    mul_x_b = 0xfffd0fa0d34c00    # Q 2.46
    y_shift = 0xfffa1f41a69800    # Q 1.47

    y = 0xfa1f4100    # Q 1.31
    ''' second '''
    x = 0xf9e68000    # Q 1.31
    b = 0x84a70600    # Q 2.30

    mul_x_b = 0x0005e0be71ce00    # Q 2.46

    x = fi(x, 1, 32, 31, 0, 'Dec')[0]
    b = fi(b, 1, 32, 30, 0, 'Dec')[0]
    mul_x_b = fi(mul_x_b, 1, 56, 46, 0, 'Dec')[0]

    print("-" * 10)
    if (x * b - mul_x_b < 10e-3):
        print(f'PASS, {mul_x_b}')

    y = 0xfa1f4100    # Q 1.31
    a = 0x84bcad00    # Q 2.30
    mul_y_a = 0x0005a90efdb5da    # Q 2.46

    y = fi(y, 1, 32, 31, 0, 'Dec')[0]
    a = fi(a, 1, 32, 30, 0, 'Dec')[0]
    mul_y_a = fi(mul_y_a, 1, 56, 46, 0, 'Dec')[0]

    print("-" * 10)
    if (y * y - mul_y_a < 10e-3):
        print(f'PASS, {mul_y_a}')

    zi_predict = mul_x_b - mul_y_a
    mul_x_b = 0x0005e0be71ce00    # Q 2.46
    mul_y_a = 0x0005a90efdb5da    # Q 2.46
    zi = mul_x_b - mul_y_a
    zi = fi(zi, 1, 56, 46, 0, 'Dec')[0]
    print("-" * 10)
    if (zi_predict - zi < 10e-3):
        print(f'PASS, {zi}')
    '''final'''
    x = 0xf9e68000    # Q 1.31
    b = 0x3dac7c00    # Q 2.30

    mul_x_b = 0xfffd0fa0d34c00    # Q 2.46

    x = fi(x, 1, 32, 31, 0, 'Dec')[0]
    b = fi(b, 1, 32, 30, 0, 'Dec')[0]
    mul_x_b = fi(mul_x_b, 1, 56, 46, 0, 'Dec')[0]

    print("-" * 10)
    if (x * b - mul_x_b < 10e-3):
        print(f'PASS, {mul_x_b}')

    y = 0xfa1f4100    # Q 1.31
    a = 0x3b6ea000    # Q 2.30
    mul_y_a = 0xfffd455378ed40    # Q 2.46

    y = fi(y, 1, 32, 31, 0, 'Dec')[0]
    a = fi(a, 1, 32, 30, 0, 'Dec')[0]
    mul_y_a = fi(mul_y_a, 1, 56, 46, 0, 'Dec')[0]

    print("-" * 10)
    if (y * a - mul_y_a < 10e-3):
        print(f'PASS, {mul_y_a}')
    else:
        print(f'FAIL, {mul_y_a}')

    zi_predict = mul_x_b - mul_y_a
    mul_x_b = 0xfffd0fa0d34c00    # Q 2.46
    mul_y_a = 0xfffd455378ed40    # Q 2.46

    zi = mul_x_b - mul_y_a
    zi = fi(zi, 1, 56, 46, 0, 'Dec')[0]
    _zi = 0xffffca4d5a5ec0
    print("-" * 10)
    if (zi_predict - zi < 10e-3):
        print(f'PASS, {zi_predict, zi}')
    else:
        print(f'FAIL, {zi_predict, zi}')

    # 0, 1, 2, 3, 4,
    gainDb = [-5912, 0]    # Q7 , int16_t
    gain = [159, 0]    # Q15, int16_t
    print(fi(gain, 1, 16, 8, 0, 'Dec'))

    gain = 0x7fff

    print(fi(gain, 1, 16, 15, 0, 'Dec'))


def iir_filter_floating_format():
    """Test for iir filter in floating format"""

    white_noise = '/Users/seunghyunoh/workplace/study/FilterDesign/ExampleMusic/music_sample.wav'
    if not os.path.exists(white_noise.split('.')[0] + '.wav'):
        cvt_pcm2wav(white_noise,
                    white_noise.split('.')[0] + '.wav', 48000, 'int16')
    white_noise = white_noise.split('.')[0] + '.wav'

    print(f'processing {white_noise}......')

    wave_processor = WaveProcessor(wavfile_path=white_noise)
    ploter = FilterAnalyzePlot(sampleing_freq=wave_processor.sampleing_freq)

    # custom filter, Q 2.30, highpass
    alpha = [0x40000000, 0x84bcadbc, 0x3b6ea04e]
    beta = [0x3dac7ca5, 0x84a706b8, 0x3dac7ca5]

    a_local = np.array(
        fi(Values=alpha,
           Signed=1,
           TotalLen=32,
           FracLen=30,
           Format=0,
           ReturnVal='Dec'))
    b_local = np.array(
        fi(Values=beta,
           Signed=1,
           TotalLen=32,
           FracLen=30,
           Format=0,
           ReturnVal='Dec'))

    print(b_local, a_local)
    wave_processor.filters = b_local, a_local
    ploter.filters = b_local, a_local

    # Plot Frequency response
    ploter.plot(name='')

    # Process test
    # outfile_path = 'test.wav'
    # result_path = ''
    # outresult_path = os.path.join(result_path, outfile_path)
    # wave_processor.run(savefile_path=outresult_path)


def filter_coff_cvt2float(fc_band):
    from lib.filt import CustomFilter
    fs = 48000
    for fc in fc_band:
        filter_custom = CustomFilter(sampling_freq=fs,
                                     cutoff_freq=fc,
                                     Qfactor=4,
                                     gain=-6)
        filter = filter_custom.peaking()
        filter = np.array(filter).tolist()
        cvt2float(filter)


def lowpass_floating_cvt2binary32():
    coeff = [
        0x5a9df7ac, 0x4026e73d, 0x2d6a8670, 0x2026f310, 0x16c310e4, 0x101d3f2e,
        0x0b68737a, 0x08138562, 0x05b7b15b, 0x040c3714, 0x02dd958b, 0x0207567b,
        0x016fa9bb, 0x01044915, 0x00b8449d, 0x008273a7, 0x005c5a50, 0x0041617a,
        0x002e493a, 0x0020c49c, 0x001732ae, 0x00106c44, 0x000ba064, 0x00083b20,
        0x0005d3bb, 0x00042011, 0x0002eba3, 0x0002114a, 0x000176b5, 0x00010946,
        0x0000bbcd, 0x000084f4, 0x00005e20, 0x000042a3, 0x00002f2d, 0x00002166,
        0x000017a5, 0x000010bd, 0x00000bda, 0x00000864, 0x000005f1, 0x00000435,
        0x000002fa, 0x0000021c, 0x0000017e, 0x0000010f, 0x000000c0, 0x00000088,
        0x00000060, 0x00000044, 0x00000031, 0x00000023, 0x00000019, 0x00000012,
        0x0000000d, 0x00000009, 0x00000007, 0x00000005, 0x00000004, 0x00000003,
        0x00000002, 0x00000002, 0x00000001, 0x00000000, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    coeff = fi(coeff, 0, 32, 31, 0, 'Dec')

    data_path = '/Users/seunghyunoh/workplace/study/FilterDesign/ExampleMusic/'
    result_path = '/Users/seunghyunoh/workplace/study/filterdesign/resultmusic/'

    infile_path = os.path.join(data_path, 'test_noise_tx_in.wav')
    fs, data = wav.read(infile_path)

    wave_processor = WaveProcessor(wavfile_path=infile_path)
    wave_processor.sampleing_freq = fs
    for c in coeff:
        wave_processor.freqfilters = c
    wave_processor.run(savefile_path=result_path +
                       '/test_noise_tx_in_lp_56.wav')


def example_2ndinterpolation():
    """Example for interpolation
    """
    from scipy.interpolate import PchipInterpolator, pchip_interpolate, CubicHermiteSpline, CubicSpline

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
        ax.plot(x, y, 'o', label='data')    # datapoint
        ax.plot(xs, np.sin(xs), label='true')    # origin
        ax.plot(xs, cs(xs), label="S")    # y
        ax.plot(xs, cs(xs, 1), label="S'")    # delta_y
        ax.plot(xs, cs(xs, 2), label="S''")    # delta_delta_y
        ax.plot(xs, cs(xs, 3), label="S'''")    # delta_delta_delta_y

    pchip_interpolation()
    # Hermite_spline_interpolation()
    plt.legend()
    plt.show()


def test_hilber_from_scratch_time_domain():
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
        print('{:2d} {:+5.2f} {:+10.2f} {:+5.2f} {:+5.2f}'.format(
            n, y[n], z1[n], z2[n], z3[n]))

    t = np.arange(0, N)

    y_fft = fft(y)

    z1_fft = fft(z1)
    z2_fft = fft(z2)
    z3_fft = fft(z3)

    fig, ax = plt.subplots(nrows=3)
    ax[0].plot(t, y, label='y')
    ax[0].plot(t, z1.imag, label='z1')
    # ax[0].plot(t, z2, label='z2')
    # ax[0].plot(t, z3, label='z3')

    ax[1].plot(t, 20 * np.log10(np.abs(y_fft)), label='y')
    ax[1].plot(t, np.abs(z1_fft), label='z1')
    # ax[1].plot(t, 20*np.log10(np.abs(z2_fft)), label='z2')
    # ax[1].plot(t, np.abs(z3_fft), label='z3')

    ax[2].plot(t, np.angle(y_fft), label='y')
    ax[2].plot(t, np.angle(z1_fft), label='z1')
    # ax[2].plot(t, np.angle(z2_fft), label='z2')
    # ax[2].plot(t, np.angle(z3_fft), label='z3')

    plt.show()


def test_hilber_from_scratch_frequency_response():
    """Test hilbert transform from frequency-domain amplitude
    """
    N = 32
    f = 1
    dt = 1.0 / N
    y = []
    for n in range(N):
        x = 2 * math.pi * f * dt * n
        y.append(2 * math.sin(x))
    y_fft = np.fft.fft(y)
    y_fft_real = np.abs(y_fft)

    # Method 1. iFFT -> disgard imag -> hilbert -> FFT
    y_fft_real_HT = y_fft_real.copy()
    y_fft_real_HT[len(y_fft_real_HT) // 2 + 1:] = 0
    y_fft_real_HT[0] /= 2
    y_fft_real_HT[len(y_fft_real_HT) // 2] /= 2
    y_HT = np.fft.ifft(y_fft_real_HT)
    y_HT = scipy.signal.hilbert(y_HT.real)
    y_fft_HT = np.fft.fft(y_HT)

    # Method 2. irFFT -> hilbert -> rFFT
    y_HT2 = np.fft.irfft(y_fft_real[:len(y_fft_real // 2 + 1)],
                         n=len(y_fft_real))
    y_HT2 = scipy.signal.hilbert(y_HT2)
    y_fft_HT2 = np.fft.rfft(y_HT2)

    x = np.arange(N)
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
    ax0.plot(y_fft_real, "*", label='fft signal')
    ax0.plot(y_fft_real_HT, "o", label='fft signal_real_Hilbert Transform')
    ax0.plot(y_fft_HT, "x", label='fft signal_Hilbert Transform')
    ax0.plot(y_fft_HT2, "s", label='rfft signal_Hilbert Transform')
    ax0.legend()

    ax1.plot(np.angle(y_fft_real, deg=True))
    ax1.plot(np.angle(y_fft_real_HT, deg=True))
    ax1.plot(np.angle(y_fft_HT, deg=True))
    ax1.plot(np.angle(y_fft_HT2, deg=True))
    fig.tight_layout()

    ax2.plot(y, "*", label='fft signal')
    ax2.plot(y_HT.imag, "o", label='fft signal_Hilbert Transform')
    ax2.plot(y_HT2.real, "s", label='rfft signal_Hilbert Transform')
    ax2.legend()
    plt.show()


def test_even_odd_cross_matrix_1d():
    """Test (5, 3, 1) <-> (5, 2) in 1D
        (5, 3, 1) * (5, 1, 2) = (5, 3, 2) 
    """
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.power(x, 2)
    print(f'array {x}, {y}')

    z = np.append(x, y, axis=-1)
    print(f'append matrix {z}')

    z = z.reshape(len(z) // len(x), len(x)).transpose().flatten()
    print(f'cross matrix {z}')

    z_2 = np.array([x, y]).flatten('F')
    print(f'cross matrix 2 {z_2}')

    z_3 = np.array([x, y]).reshape((len(x) + len(y),), order='F')
    print(f'cross matrix 3 {z_3}')


def test_even_odd_cross_matrix_2d():
    """Test (5, 3, 1) <-> (5, 2) in 2D
        (5, 3, 1) * (5, 1, 2) = (5, 3, 2) 
    """
    print('\nBasic Test')
    x = np.array([np.arange(0, 16) for _ in range(32)])
    x = np.expand_dims(x, axis=-1)
    y = np.ones(shape=(32, 3))
    y[:, 0] = 0
    y[:, 2] = 2
    y = np.expand_dims(y, axis=1)
    z = np.matmul(x, y)
    print(f'{x.shape} @ {y.shape} = {z.shape}')
    # for i in range(16): print(f'{x[0, i, :]} * {y[i, :]} = {z[0, i, :]}')
    z = z.reshape((z.shape[0], z.shape[1] * z.shape[2]))
    print(z[0])

    print('\nTest graphic equalizer')
    y = np.array([np.zeros((32,)),
                  np.ones((32,)),
                  np.ones((32,)) * 2]).transpose()
    y = np.expand_dims(y, axis=1)
    z = np.matmul(x, y)
    print(f'{x.shape} @ {y.shape} = {z.shape}')

    # for i in range(16): print(f'{x[0, i, :]} * {y[i, :]} = {z[0, i, :]}')

    z = z.reshape((z.shape[0], z.shape[1] * z.shape[2]))
    z = np.append(z, np.ones((z.shape[0], 1)) * 1000, axis=1)
    print(z[0])


def iir_filter_mulitstage_plot_parallel(file):
    """Test the frequency response of parallel structure of iir filter 
    """
    # set the target
    data_path = '/Users/seunghyunoh/workplace/study/FilterDesign/ExampleMusic/'
    infile_path = os.path.join(data_path, 'test_noise_tx_in.wav')
    fs, data = wav.read(infile_path)

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

    # set the gain - gain_c1, gain_c2
    # set the fc type - fc, fc_twice
    target_gain = gain
    target_fc = fc

    eq_set = GraphicalEqualizer(fs, target_fc, target_gain, Q)
    eq_set.freqz()

    eq_set.write_to_file(file)


def iir_filter_multistage_process_parallel(file):
    """Test wav file processing of parallel structure of iir filter 
    """
    data_path = '/Users/seunghyunoh/workplace/study/FilterDesign/ExampleMusic/'
    result_path = '/Users/seunghyunoh/workplace/study/filterdesign/resultmusic/'

    infile_path = os.path.join(data_path, 'White Noise.wav')
    fs, data = wav.read(infile_path)
    wave_processor = WaveProcessor(wavfile_path=infile_path)
    wave_processor.graphical_equalizer = True

    coeff_text = open(file).read()
    coeff_text = coeff_text.split('\n')[:-1]
    coeff_text = [text.split(' ') for text in coeff_text]
    char2num(coeff_text)

    coeff_text, bias = np.array(coeff_text[:-1],
                                dtype=np.float64), np.array(coeff_text[-1])
    coeff = [coeff_text, bias]

    wave_processor.filters = coeff

    outresult_path = result_path + '/whitenoise' + file.split('.')[-2].split(
        '/')[-1] + '.wav'
    print(f'file {outresult_path} is processing......')

    wave_processor.run(savefile_path=outresult_path)


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

    x = graphs[1][0]
    y = graphs[1][1] - graphs[0][1]

    plt.plot(x, y)
    plt.xscale('log')
    plt.title('Frequency Response', fontsize=14)
    plt.ylabel('Amplitude [dB]')
    plt.xlabel('Frequency [Hz]')
    plt.grid(which='both', linestyle='-', color='grey')
    plt.xticks(
        [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
        ["10", "50", "100", "200", "500", "1K", "2K", "5K", "10K", "20K"])

    plt.show()


if __name__ == '__main__':
    logger = maker_logger()
    logger.info("Hello Digital Signal Processing World!")

    fc_band = np.array([250, 2000, 8000])
    """ Several types of filters and
        Analysis analog and digital scope of pole and zeros """
    # plot_cascade_sheving_filter()
    # plot_sheving_filter_digital()
    # plot_sheving_filter_analog()
    """Cascade of filters coffcient design"""
    iir_filter_mulitstage_plot_cascade(fc_band)
    # iir_filter_mulitstage_process_cascade(fc_band)
    """ fi command example """
    # fi_example()
    """ RTK floating <-> binary floating format parameter test """
    # iir_filter_floating_format()
    # filter_coff_cvt2float(fc_band)
    # lowpass_floating_cvt2binary32()
    """ Non-linear, linear interpolation test """
    # example_2ndinterpolation()
    """ Hilbert Transform test """
    # test_hilber_from_scratch_time_domain()
    # test_hilber_from_scratch_frequency_response()
    """ Matrix Transform test"""
    # test_even_odd_cross_matrix_1d()
    # test_even_odd_cross_matrix_2d()
    """Parallel of filters coffcient design"""
    file = os.path.join(ROOT, 'lib/data/coeff_test.txt')
    # iir_filter_mulitstage_plot_parallel(file)
    # iir_filter_multistage_process_parallel(file)
    """ Audacity Frequency Response"""
    # plot_audacity_freq_response()
