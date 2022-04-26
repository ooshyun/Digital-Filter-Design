"""[TODO]
Test for filter design
    This is testing file for filt.py, filter_analyze.py, filter_application.py,
    graphical_equalizer.py, fi.py which is on this projects. The entire object is
    Filter Design for Equalizer on the device.

    The types of tests is like below.
    - Frequency response for several types of filters
    - Floating to binary format using fi command
    - Cascade of filters's plot and processing
    - Parallel of filters's plot and processing
"""
import os
import json
import numpy as np
from numpy import log10, pi, sqrt
import scipy.io.wavfile as wav
from scipy.fftpack import *

from src import (
    FilterAnalyzePlot,
    WaveProcessor,
    ParametricEqualizer,
    GraphicalEqualizer,
    cvt_char2num,
    maker_logger,
    DEBUG,
)

if DEBUG:
    PRINTER = maker_logger()

ROOT = os.getcwd()


def serial_equalizer_plot():
    """Test frequency response for IIR filter cascade
    """
    from src import peaking

    data_path = "./test/data/wav/"
    infile_path = os.path.join(data_path, "White Noise.wav")
    fs, _ = wav.read(infile_path)

    ploter = FilterAnalyzePlot()
    parametric_filter = ParametricEqualizer(fs)
    fc_band = np.array([250, 2000, 8000])

    for f in fc_band:
        peak_filter = peaking(Wn=2 * f / fs, dBgain=-6, Q=4)
        parametric_filter.coeff = peak_filter

    ploter.filters = parametric_filter
    ploter.plot(type=["freq", "phase", "pole"])


def serial_equalizer_process():
    """Test processing to wav for IIR filter cascade
    """
    from src import peaking

    data_path = "./test/data/wav/"
    result_path = "./test/result/wav/"
    infile_path = os.path.join(data_path, "White Noise.wav")
    fs, _ = wav.read(infile_path)

    wave_processor = WaveProcessor(wavfile_path=infile_path)
    fc_band = np.array([250, 2000, 8000])

    for f in fc_band:
        peak_filter = peaking(Wn=2 * f / fs, dBgain=6, Q=4)
        b, a = peak_filter
        wave_processor.filters = b, a

    # wave_processor.graphical_equalizer = True
    wave_processor.run(
        savefile_path=result_path + "/whitenoise_3peak_250_2000_8000.wav"
    )

    print(sum(wave_processor.freqfilter_time) / len(wave_processor.freqfilter_time))
    print(sum(wave_processor.timefilter_time) / len(wave_processor.timefilter_time))


def parallel_equalizer_plot():
    with open("./test/data/json/test_graphical_equalizer.json", "r") as f:
        test_case = json.load(f)

    fs, fc, gain = (
        test_case["1"]["sampling_frequency"],
        test_case["1"]["cutoff_frequency"],
        test_case["1"]["test_gain"][0],
    )

    fs = int(fs)
    fc = np.array(fc)
    gain = np.array(gain)

    eq = GraphicalEqualizer(fs, fc, gain)
    w, h = eq.freqz(show=True)

    file = "/test/data/txt/test_graphical_equalizer.txt"
    eq.write_to_file(f"{ROOT}/{file}")


def parallel_equalizer_wav_process():
    with open("./test/data/json/test_graphical_equalizer.json", "r") as f:
        test_case = json.load(f)

    fs, fc, gain = (
        test_case["1"]["sampling_frequency"],
        test_case["1"]["cutoff_frequency"],
        test_case["1"]["test_gain"][0],
    )

    fs = int(fs)
    fc = np.array(fc)
    gain = np.array(gain)

    eq = GraphicalEqualizer(fs, fc, gain)
    w, h = eq.freqz(show=True)

    txt_file = "/test/data/txt/test_graphical_equalizer.txt"
    eq.write_to_file(f"{ROOT}/{txt_file}")

    """Test wav file processing of parallel structure of iir filter 
    """
    data_path = "./test/data/wav/"
    result_path = "./test/result/wav/"

    wav_file = "White Noise.wav"
    out_file = "White Noise_graphical_equalizer.wav"
    infile_path = os.path.join(data_path, wav_file)
    outfile_path = os.path.join(result_path, out_file)

    coeff_text = open(f"{ROOT}/{txt_file}").read()
    coeff_text = coeff_text.split("\n")[:-1]
    coeff_text = [text.split(" ") for text in coeff_text]
    cvt_char2num(coeff_text)

    coeff_text, bias = np.array(coeff_text[:-1]), np.array(coeff_text[-1])

    wave_processor = WaveProcessor(wavfile_path=infile_path)
    wave_processor.graphical_equalizer = True
    wave_processor.filters = coeff_text
    wave_processor.bias = bias

    outresult_path = outfile_path

    PRINTER.info(f"target file {outresult_path} is processing......")
    wave_processor.run(savefile_path=outresult_path)


def filter_plot():
    from src import lowpass, highpass, bandpass, notch, peaking, shelf, allpass

    curr_path = os.getcwd()
    data_path = os.path.join(curr_path, "test/data/wav/")
    file_name = "White Noise.wav"
    result_path = ""

    infile_path = os.path.join(data_path, file_name)
    fs, data = wav.read(infile_path)

    fft_size = 256
    fft_band = np.arange(1, fft_size / 2 + 1) * fs / fft_size
    # fc_band = np.arange(30, 22060, 10)
    fc_band = np.array([100, 1000, 2000, 3000, 5000])

    ploter = FilterAnalyzePlot(sampling_freq=fs)

    """Plot the several filters
    """
    fc = 1000
    gain = 3
    Q = 1 / np.sqrt(2)
    name = "Shelf Filter"

    lowpass_filter = lowpass(Wn=2 * fc / fs, Q=Q)
    highpass_filter = highpass(Wn=2 * fc / fs, Q=Q)
    bandpass_filter = bandpass(Wn=2 * fc / fs, Q=Q)
    notch_filter = notch(Wn=2 * fc / fs, Q=Q)
    peak_filter = peaking(Wn=2 * fc / fs, Q=Q, dBgain=gain)
    shelf_filter = shelf(Wn=2 * fc / fs, Q=Q, dBgain=gain)
    allpass_filter = allpass(Wn=2 * fc / fs, Q=Q)

    ploter.filters = shelf_filter
    ploter.plot(type=["freq", "phase", "pole"], save_path=None, name=name)


def filter_process():
    """Comparison between time domain and frequency domain using WavProcessor class
    """
    from src import peaking

    curr_path = os.getcwd()
    data_path = os.path.join(curr_path, "test/data/wav/")
    file_name = "White Noise.wav"

    outfile_path = "./test/result/wav/"

    infile_path = os.path.join(data_path, file_name)
    fs, data = wav.read(infile_path)

    fc = 1033.59375
    # time
    wave_processor = WaveProcessor(wavfile_path=infile_path)
    outfile_name = "White Noise_peaking_time_domain.wav"
    peak_filter = peaking(Wn=2 * fc / fs, Q=1 / np.sqrt(2), dBgain=3)
    wave_processor.filters = peak_filter
    wave_processor.run(savefile_path=outfile_path + outfile_name)
    print(sum(wave_processor.timefilter_time) / len(wave_processor.timefilter_time))

    # frequency
    wave_processor = WaveProcessor(wavfile_path=infile_path)
    outfile_name = "White Noise_peaking_freq_domain.wav"

    fft_size = 256
    fft_band = np.arange(1, fft_size / 2 + 1) * fs / fft_size
    # fc_band = np.arange(30, 22060, 10)
    fc_band = np.array([100, 1000, 2000, 3000, 5000])
    idfreq = np.argwhere(fft_band == fc)
    wave_processor.freqfilters = idfreq
    wave_processor.run(savefile_path=outfile_path + outfile_name)

    print(sum(wave_processor.freqfilter_time) / len(wave_processor.freqfilter_time))


def analyze_filter():
    from src import highpass, notch

    fs = 44100

    # """ Custom filter analysis"""
    ploter = FilterAnalyzePlot(sampling_freq=44100)

    fc = 1000
    filter_custom = highpass(Wn=2 * fc / fs, Q=1 / np.sqrt(2))

    ploter.filters = filter_custom
    ploter.plot(type=['freq', 'phase', 'pole'])

    del filter_custom

    """ Parametric filter analysis"""
    ploter = FilterAnalyzePlot()

    fc = np.array([500, 4000])

    peq = ParametricEqualizer(fs)

    filter_custom = notch(Wn=2 * fc[0] / fs, Q=1 / np.sqrt(2))
    peq.coeff = filter_custom

    filter_custom = notch(Wn=2 * fc[1] / fs, Q=1 / np.sqrt(2))
    peq.coeff = filter_custom

    ploter.filters = peq
    ploter.plot(type=['freq', 'phase', 'pole'])

    del peq

    """ Graphical filter analysis"""
    with open("./test/data/json/test_graphical_equalizer.json", "r") as f:
        test_case = json.load(f)

    fs, fc, gain = (
        test_case["1"]["sampling_frequency"],
        test_case["1"]["cutoff_frequency"],
        test_case["1"]["test_gain"][0],
    )

    fs = int(fs)
    fc = np.array(fc)
    gain = np.array(gain)

    ploter = FilterAnalyzePlot()
    geq = GraphicalEqualizer(fs, fc, gain)

    ploter.filters = geq
    ploter.plot(type=["freq", "phase", "pole"])

    del geq

    del ploter


if __name__ == "__main__":
    PRINTER.info("Hello Digital Signal Processing World!")

    """filter cofficient design"""
    # filter_plot()
    # filter_process()

    """Serial of filters coffcient design"""
    # serial_equalizer_plot()
    # serial_equalizer_process()

    """Parallel of filters coffcient design"""
    # parallel_equalizer_plot()
    # parallel_equalizer_wav_process()

    # analyze_filter()
    pass
