import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def example_1():
    # https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    # https://ratsgo.github.io/speechbook/docs/fe/mfcc#log-mel-spectrum
    sample_rate, signal = wav.read('./study/mel-spectrogram/OSR_us_000_0010_8k.wav')  # File assumed to be in the same directory
    signal = signal[0:int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    frame_size = 0.025
    frame_stride = 0.01
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
            np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    frames *= np.array([0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1)) for n in range(frame_length)])

    NFFT = 512
    dft_frames = np.fft.rfft(frames, NFFT)
    mag_frames = np.absolute(dft_frames)
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    print(fbank.shape)

    for bank in fbank:
        plt.plot(bank, )#".")
    plt.show()

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability

    print(filter_banks.shape)

    filter_banks = 20 * np.log10(filter_banks)  # dB

def example_2():
    """
    Mel basis, hz_to_mel
    -----------

    htk: 
        2595 * log10(1 + frequency/700)
    else:
        f_sp = 200.0 / 3

        min_log_hz = 1000
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0 

        1. frequencies < 1000
            mel = 3*frqeuency / 200

        2. frequencies >= 1000
            
            logstep = log2(6.4) / 27.0
            mel = min_log_mel + np.log(frequencies / min_log_hz) / logstep
                = 3*(1000-0) / 200 + log2(frequencies / 1000) / logstep
                = 15 + logstep * log2(frequencies / 1000)
                = 15 + 27.0 * log_{6.4}(frequencies / 1000)
    
    Mel basis
    -----------
    librosa method explanation: ./study/mel-spectrogram/mel.jpg
    
    Goal: f_{mel_k} = a_{(0,k)}f_{(0,k)} + a_{(1,k)}f_{(1,k)} +  \dots + a_{(n,k)}f_{(n,k)}

    1. Hz -> Frequencies
    2. Using np.diff and np.subtract.outer, getting below equation,

    H_m(k) = \begin{cases}
                0 & k<f(m-1) \\
                
                - \dfrac{f(m-1)-k}{f(m)-f(m-1)} & f(m-1) \leq k \leq f(m) \\
                
                \dfrac{f(m+1)-k}{f(m+1)-f(m)}& f(m) \leq k \leq f(m+1) \\
                
                0 & k > f(m+1)
            \end{cases}

    3. Normalization
        - norm l1 l2
        - slaney, 2/(mel_f_n+2 - mel_f_n) * weights
    """
    wav, sr = librosa.load(librosa.ex("trumpet"))
    n_fft = 2048
    hop_length = 512
    n_mels = 40
    power = 2.

    # librosa Mel Spectrogram
    melspec = librosa.feature.melspectrogram(wav, 
                                    sr=sr, 
                                    n_mels=n_mels,
                                    n_fft=n_fft, 
                                    hop_length=hop_length, 
                                    win_length=n_fft, 
                                    window="hann", 
                                    center=True, 
                                    pad_mode = "constant", 
                                    power=power,
                                    # htk=True,
                                    )

    # Scratch Mel Spectrogram
    S = librosa.stft(y=wav, 
                    n_fft=n_fft, 
                    hop_length=hop_length, 
                    win_length=n_fft, 
                    window="hann", 
                    center=True, 
                    pad_mode="constant")

    S_spectrogram = np.abs(S) ** power

    mel_basis = librosa.filters.mel(sr=sr, 
                                n_fft=n_fft, 
                                n_mels=n_mels, 
                                htk=False
                                )

    mel_basis_htk = librosa.filters.mel(sr=sr, 
                                n_fft=n_fft, 
                                n_mels=n_mels, 
                                htk=True
                                )

    melspec_scratch = mel_basis @ S_spectrogram
    melspec_htk_scractch = mel_basis_htk @ S_spectrogram
    
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    img0 = librosa.display.specshow(mel_basis_htk, x_axis='log', ax=ax0)
    img1 = librosa.display.specshow(mel_basis, x_axis='log', ax=ax1)
    ax0.set(ylabel='Mel filter', title='htk Mel filter bank')
    ax1.set(ylabel='Mel filter', title='Mel filter bank')
    fig.colorbar(img0, ax=ax0)
    fig.colorbar(img1, ax=ax1)

    # fig, (ax0, ax1) = plt.subplots(nrows=2)
    # S_dB = librosa.power_to_db(melspec, ref=np.max)
    # S_dB_scratch = librosa.power_to_db(melspec_scratch, ref=np.max)
    # S_dB_htk_scratch = librosa.power_to_db(melspec_htk_scractch, ref=np.max)
    # # S_dB_scratch = 10*np.log10(melspec_scratch)
    
    # img = librosa.display.specshow(S_dB_scratch, x_axis='time',
    #                           y_axis='mel', sr=sr,
    #                           fmax=8000, ax=ax0)

    # img = librosa.display.specshow(S_dB_htk_scratch, x_axis='time',
    #                           y_axis='mel', sr=sr,
    #                           fmax=8000, ax=ax1)    

    # fig.colorbar(img, ax=ax0, format='%+2.0f dB')
    # fig.colorbar(img, ax=ax1, format='%+2.0f dB')

    # ax0.set(title='Mel-frequency spectrogram (Slaney)')
    # ax1.set(title='Mel-frequency spectrogram (htk)')

    plt.show()

if __name__ == "__main__":
    # example_1()
    example_2()