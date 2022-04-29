"""Wav source generator
    It contains sine tone, white noise and increasing tone generator.

    TODO LIST
    ---------
    - Pink noise
    - Brown noise
    - improved White noise 
"""
import wave
import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from scipy.io import wavfile as wav
from scipy.signal import chirp, sweep_poly, spectrogram

import librosa, librosa.display


def plot_fft(data: np.array, framesize: int, sample_rate: int) -> None:
    """Plot for frequency domain

    Args:
      data: the data to plot in frequency domain
      framesize: the block size divided by sample frequency
      samplingrate

    Returns:

    Raises:
    """
    bins = np.arange(0, 10000) * sample_rate / 10000
    y_fft = np.fft.fft(data[:framesize])
    plt.plot(bins[:math.trunc(sample_rate / 2) + 1],
             abs(y_fft[:math.trunc(sample_rate / 2) + 1]))
    plt.show()


def _get_random(seeds, size):
    """Get random number array

    """

    if len(seeds) > 3:
        return 0
    array = []
    seed_count = 0
    gama = 0
    for i in range(size):
        val = bin(seeds[seed_count]**2)
        val = val[2:]
        if len(val) > 32:
            val = val[(len(val) - 32):]
        val = val.zfill(32)
        val = val[0 + gama:16 + gama]
        val = int(val, 2)
        if gama == 15:
            gama = 0
        else:
            gama += 1
        if seed_count == 2:
            seeds[seed_count] = val + seeds[0]
            seed_count = 0
        else:
            seeds[seed_count] = val + seeds[seed_count + 1]
            seed_count += 1
        array.append(val)

    return np.array(array)


def create_whitenoise():
    """White Noise  
        Random number generation with 16 bit variable
    - Using 3 seeds
    - Just 2 counters
    - Only one squared, one addition operation and one shift
    - Only 10 bytes need for RAM

                            THE ALGORITHM

    1- Take the seed using counter to select between the 3. The counter is
    increased by 1 every loop. Here is an example of 3 seeds.

    seeds=[65321,12043,2769]

    val=seeds[seed_counter]=65321= 1111111100101001

    2- Square the current "val"  value to get twice the size (32 bit now)

    val=val*val= 11111110010100101011010010010001

    3- Select the output based on the gama factor. Gama, same as seed counter,
    gets updated every loop, being that the maximum is the number of bits
    precision, this case 16 (0 to 15)

    11111110010100101011010010010001
    |     gama=0    |
    |     gama=1    |
    |     gama=2    |
                ...
                |     gama=15    |

    For gama=0, the output value will be 1111111001010010 or 65106

    4- Next step is to update the seeds. Using the output value,
    add this value to the next seed. Hence the values will be
    even more likely to immitate random behaviour.

    seeds[seed_counter]=val+seeds[seed_counter+1]

    For seed_counter=0

    seeds[0]=65106+12043= 11613 = 0010110101011101*

    * Note that the value overflew and there is no
    problem with overflow

    """
    seeds = [65321, 12043, 2769]    # Seeds (1,2 and 3)
    size = 160000    # Output samples for the random code
    Original_array = _get_random(seeds, size)
    print(type(Original_array), Original_array.dtype)
    wav.write('./whiteNoise.wav', 16000, Original_array)


def create_sinetone(frequency):
    """Create sine tone

    Parameters
    ----------
      frequency

    Returns
    -------
        None
    """
    fs = 44100
    total_time = 10
    save_sinewav = 0

    t = np.arange(0, total_time * fs)
    y = np.sin(2 * np.pi * frequency * t / fs)
    y = np.float32(y)
    # Analyze in spectrum
    plot_fft(y, framesize=10000, fs=fs)

    return fs, y


def create_increase_sine_tone(start=125, end=20000, samplingrate=44100):
    """Create sine tone

    Args:
      start: 
        Frequency to get the wave in first period
      end: 
        Frequency to get the wave in last period

    Returns:
        frequency_sampling: 
            Sampling Frequency
        wave: 
            Wave increasing tone from start to end frequency
    Raises:
        None
    """
    frequency_sampling = samplingrate
    period = 10
    time = np.arange(0, int(period * frequency_sampling)) / frequency_sampling
    wave = chirp(time, f0=start, f1=end, t1=period, method='linear')
    return frequency_sampling, wave


def analyze_stft(samplingrate, wave, style="specshow"):
    """Plot frequency component in the wave having fast change between frequencys 

    Args:
      samplingrate: 
        sampling frequency
      wave: 
        data to plot
      style: {'specshow', 'line'}, optional
        If not given, 'specshow' is assumed.

        specshow: 
            spectogram style
        
        line:
            line style

    Returns:
        None
    Raises:
        None
    """

    def _normalize(S, minleveldb):
        return np.clip((S - minleveldb) / (-minleveldb), 0, 1)

    fft_size = 1024
    hop_length = 256
    win_length = 1024
    # audio_sample, fs = librosa.load(
    #     "/Users/seunghyunoh/workplace/Project/OliveUnion/RealtimeDSP/GUI_Compression/data/sin20000-250.wav", sr=None)
    sdata = librosa.stft(wave,
                         n_fft=fft_size,
                         hop_length=hop_length,
                         win_length=win_length,
                         window=signal.hann)
    S = np.abs(sdata)

    # FFT -> plot
    # normalize_function
    min_level_db = -100

    mag_db = librosa.amplitude_to_db(S)
    mag_n = _normalize(mag_db, min_level_db)

    if style == "specshow":
        plt.plot()
        librosa.display.specshow(mag_n, y_axis='linear', x_axis='time', sr=fs)
        plt.title('spectrogram')

    if style == 'line':
        t = np.linspace(0, 24000, mag_db.shape[0])
        plt.plot(t, mag_db[:, 100].T)
        plt.title('magnitude (dB)')
    plt.show()


if __name__ == '__main__':
    """create 5k sine tone"""
    # frequency = 5000
    # fs, wave = create_sinetone(frequency)
    #  wav.write('./sin' + str(frequency) + '.wav', fs, wave)

    """create 250-20k sine tone"""
    start, end = 250, 20000
    fs, wave = create_increase_sine_tone(start=start, end=end)
    analyze_stft(fs, wave)
    # wav.write('./sin' + str(start)+'-'+str(end)+'.wav', fs, wave)
