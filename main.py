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

    ploter = FilterAnalyzePlot(sample_rate=fs)

    """Plot the several filters
    """
    fc = 1033.59375
    gain = 6
    Q = 1 / np.sqrt(2)
    name = "Shelf Filter"

    lowpass_filter = lowpass(Wn=2 * fc / fs, Q=Q)
    highpass_filter = highpass(Wn=2 * fc / fs, Q=Q)
    bandpass_filter = bandpass(Wn=2 * fc / fs, Q=Q)
    notch_filter = notch(Wn=2 * fc / fs, Q=Q)
    peak_filter = peaking(Wn=2 * fc / fs, Q=Q, dBgain=gain)
    shelf_filter = shelf(Wn=2 * fc / fs, Q=Q, dBgain=gain)
    allpass_filter = allpass(Wn=2 * fc / fs, Q=Q)

    ploter.filters = peak_filter
    ploter.plot(type=["freq", "phase", "pole"], save_path=None, name=name)


def filter_process():
    """Comparison between time domain and frequency domain using WavProcessor class
    """
    from src import peaking, shelf

    curr_path = os.getcwd()
    data_path = os.path.join(curr_path, "test/data/wav/")
    file_name = "White Noise.wav"

    outfile_path = "./test/result/wav/"

    infile_path = os.path.join(data_path, file_name)
    fs, data = wav.read(infile_path)

    gain = 6
    fc = 1033.59375
    # time
    wave_processor = WaveProcessor(wavfile_path=infile_path)
    outfile_name = "White Noise_peak_time_domain.wav"
    peak_filter = peaking(Wn=2 * fc / fs, Q=1 / np.sqrt(2), dBgain=gain)
    wave_processor.filter_time_domain_list = peak_filter
    wave_processor.run(savefile_path=outfile_path + outfile_name)

    if len(wave_processor.time_filter_time) != 0:
        print(
            sum(wave_processor.time_filter_time) / len(wave_processor.time_filter_time)
        )

    # frequency
    wave_processor = WaveProcessor(wavfile_path=infile_path)
    outfile_name = "White Noise_peaking_freq_domain.wav"

    fft_size = 256  # it should be designed before running
    fft_band = np.arange(1, fft_size // 2 + 1) * fs / fft_size
    coeff_frequency = np.ones(shape=(fft_size // 2 + 1,))
    coeff_frequency[np.argwhere(fft_band == fc)] = 10 ** (gain / 20)
    wave_processor.filter_freq_domain_list = coeff_frequency
    wave_processor.run(savefile_path=outfile_path + outfile_name)

    if len(wave_processor.time_filter_freq) != 0:
        print(
            sum(wave_processor.time_filter_freq) / len(wave_processor.time_filter_freq)
        )


def serial_equalizer_plot():
    """Test frequency response for IIR filter cascade
    """
    from src import peaking

    data_path = "./test/data/wav/"
    infile_path = os.path.join(data_path, "White Noise.wav")
    fs, _ = wav.read(infile_path)

    ploter = FilterAnalyzePlot()
    parametric_filter = ParametricEqualizer(fs)
    fc_band = np.array([1000, 4000, 8000])

    for f in fc_band:
        peak_filter = peaking(Wn=2 * f / fs, dBgain=6, Q=4)
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
    fc_band = np.array([1000, 4000, 8000])

    for f in fc_band:
        peak_filter = peaking(Wn=2 * f / fs, dBgain=12, Q=4)
        b, a = peak_filter
        wave_processor.filter_time_domain_list = b, a

    # wave_processor.graphical_equalizer = True
    wave_processor.run(
        savefile_path=result_path + "/whitenoise_3peak_250_2000_8000.wav"
    )

    if len(wave_processor.time_filter_freq) != 0:
        print(
            sum(wave_processor.time_filter_freq) / len(wave_processor.time_filter_freq)
        )
    if len(wave_processor.time_filter_time) != 0:
        print(
            sum(wave_processor.time_filter_time) / len(wave_processor.time_filter_time)
        )


def generator_test_vector_grahpical_equalizer():
    """Generate test vector for parallel strucuture equalizer called graphical equalizer
    """
    sample_rate = 44100

    # cuf-off freuqency case 1
    cutoff_frequency = np.array(
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

    # gain
    num_case = 5
    test_gain_list = np.zeros(shape=(num_case, len(cutoff_frequency)))

    # case 1
    test_gain_list[0, :] = np.array(
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

    # case 2
    test_gain_list[1, 0::2] = 12
    test_gain_list[1, 1::2] = -12

    # case 3
    test_gain_list[2, np.where(cutoff_frequency == 2000)] = 12

    # case 4
    test_gain_list[3, :] = np.ones_like(cutoff_frequency) * 12

    # case 5
    test_gain_list[4, 0::3] = 0
    test_gain_list[4, 1::3] = 0
    test_gain_list[4, 2::3] = 12

    # cut-off frequency case 2, cutoff frequency with bandwith
    f_bandwidth = np.array(
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
    f_upperband = np.array(
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
    f_lowerband = np.zeros_like(f_upperband)
    f_lowerband[0] = 17.5
    f_lowerband[1:] = f_upperband[:-1]

    cutoff_frequency_bandwidth = np.zeros((2, len(cutoff_frequency)))
    cutoff_frequency_bandwidth[0, :] = np.append(10, f_upperband[:-1])
    cutoff_frequency_bandwidth[1, :] = cutoff_frequency
    cutoff_frequency_bandwidth = cutoff_frequency_bandwidth.reshape(
        (cutoff_frequency_bandwidth.shape[0] * cutoff_frequency_bandwidth.shape[1],),
        order="F",
    )

    test_gain_bandwidth_list = np.zeros(
        shape=(num_case, cutoff_frequency_bandwidth.shape[0])
    )

    for id_test_gain, test_gain in enumerate(test_gain_list):
        buf_test_gain = np.zeros((2, len(cutoff_frequency)))
        buf_test_gain[0, :] = test_gain
        buf_test_gain[1, :] = test_gain
        buf_test_gain = buf_test_gain.reshape(
            (buf_test_gain.shape[0] * buf_test_gain.shape[1],), order="F"
        )
        buf_test_gain[1:] = buf_test_gain[:-1]
        buf_test_gain[0] = 0

        test_gain_bandwidth_list[id_test_gain, :] = buf_test_gain[:]

    cutoff_frequency = cutoff_frequency.tolist()
    test_gain_list = test_gain_list.tolist()

    cutoff_frequency_bandwidth = cutoff_frequency_bandwidth.tolist()
    test_gain_bandwidth_list = test_gain_bandwidth_list.tolist()

    test_vector_graphical_equalizer = json.dumps(
        {
            "1": {
                "sample_rate": sample_rate,
                "cutoff_frequency": cutoff_frequency,
                "test_gain": test_gain_list,
            },
            "2": {
                "sample_rate": sample_rate,
                "cutoff_frequency": cutoff_frequency_bandwidth,
                "test_gain": test_gain_bandwidth_list,
            },
        },
        indent=4,
    )
    with open("./test/data/json/test_graphical_equalizer.json", "w") as f:
        f.write(test_vector_graphical_equalizer)


def parallel_equalizer_plot():
    with open("./test/data/json/test_graphical_equalizer.json", "r") as f:
        test_case = json.load(f)

    fs, fc, gain = (
        test_case["2"]["sample_rate"],
        test_case["2"]["cutoff_frequency"],
        test_case["2"]["test_gain"][1],
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
        test_case["2"]["sample_rate"],
        test_case["2"]["cutoff_frequency"],
        test_case["2"]["test_gain"][1],
    )

    fs = int(fs)
    fc = np.array(fc)
    gain = np.array(gain)

    eq = GraphicalEqualizer(fs, fc, gain)
    # w, h = eq.freqz(show=True)

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
    wave_processor.filter_time_domain_list = coeff_text
    wave_processor.bias = bias

    outresult_path = outfile_path

    PRINTER.info(f"target file {outresult_path} is processing......")
    wave_processor.run(savefile_path=outresult_path)


def analyze_filter():
    from src import highpass, notch

    fs = 44100

    # """ Custom filter analysis"""
    ploter = FilterAnalyzePlot(sample_rate=44100)

    fc = 1000
    filter_custom = highpass(Wn=2 * fc / fs, Q=1 / np.sqrt(2))

    ploter.filters = filter_custom
    ploter.plot(type=["freq", "phase", "pole"])

    del filter_custom

    """ Parametric filter analysis, serial structure"""
    ploter = FilterAnalyzePlot()

    fc = np.array([500, 4000])

    peq = ParametricEqualizer(fs)

    filter_custom = notch(Wn=2 * fc[0] / fs, Q=1 / np.sqrt(2))
    peq.coeff = filter_custom

    filter_custom = notch(Wn=2 * fc[1] / fs, Q=1 / np.sqrt(2))
    peq.coeff = filter_custom

    ploter.filters = peq
    ploter.plot(type=["freq", "phase", "pole"])

    del peq

    """ Graphical filter analysis, parallel structure"""
    with open("./test/data/json/test_graphical_equalizer.json", "r") as f:
        test_case = json.load(f)

    fs, fc, gain = (
        test_case["1"]["sample_rate"],
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

    """Single filter design"""
    # filter_plot()
    # filter_process()

    """Serial structure of filters design"""
    # serial_equalizer_plot()
    # serial_equalizer_process()

    """Parallel structure of filters design"""
    # generator_test_vector_grahpical_equalizer()
    # parallel_equalizer_plot()
    # parallel_equalizer_wav_process()

    """ Analyze filter"""
    # analyze_filter()
    pass
