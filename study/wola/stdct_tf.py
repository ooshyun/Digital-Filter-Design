import numpy as np
import tensorflow as tf

def frame(signals, frame_length, frame_step, winfunc=tf.signal.hamming_window):
    framed_signals = tf.signal.frame(signals, frame_length, frame_step, pad_end=False)
    if winfunc is not None:
        window = winfunc(frame_length, dtype=tf.float32)
        framed_signals *= window
    return framed_signals

def over_lap_and_add(framed_signals, frame_length, frame_step, winfunc=tf.signal.hamming_window):
  """overlap and add
  params：
    framed_signals: tf.float32, shape=[batch, n_frames, frame_length]
    frame_length: Window length
    frame_step: frame shift
  return：
    signals: tf.float32, shape=[batch, x_length]
  """
  shape = tf.shape(framed_signals)
  n_frames = shape[1]
  # Generate de-overlapping windows
  if winfunc is not None:
    window = winfunc(frame_length, dtype=tf.float32)
    window = tf.reshape(window, [1, frame_length])
    window = tf.tile(window, [n_frames, 1])
    window = tf.signal.overlap_and_add(window, frame_step)
  signals = tf.signal.overlap_and_add(framed_signals, frame_step)
  signals /= window
  signals = tf.cast(signals, tf.float32)
  return signals

def Analysis(signals, frame_length, frame_step):
  """
  signals: shape=[batch, height]
  return
    spec_real: tf.float32, shape=[batch, n_frames, fft_length]"""
  with tf.name_scope("Analysis"):
    # frame
    framed_signals = frame(signals, frame_length, frame_step)
    # DFT
    spec = tf.signal.dct(framed_signals, type=2, norm='ortho')

    spec_real = tf.cast(spec, tf.float32)
    spec_real = spec_real.numpy()

  return spec_real


if __name__=="__main__":
    import librosa
    import matplotlib.pyplot as plt
    sample_rate = 16000
    nfft = 1024
    hop_length = 256
    sig, sr = librosa.load("[TODO] PATH", sr=sample_rate)
    tfsignal = Analysis(sig, 1024, 256)
    tfsignal.shape
    tfsignal = tfsignal.transpose()

    magnitude = np.abs(tfsignal)

    log_spectrogram = librosa.amplitude_to_db(magnitude)

    plt.figure()
    librosa.display.specshow(log_spectrogram, sr=sr, vmin=-40, vmax=30, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.title("DCT Spectrogram (dB)")