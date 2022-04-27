from random import sample
import os
import scipy.io.wavfile as wav
import librosa
import soundfile as sf
import numpy as np

"""
WavFileWarning: Chunk (non-data) not understood, skipping it.

"""


def read_wav(file_path: str):
    data, samplerate = librosa.load(file_path, sr=44100, mono=False)
    if data.dtype == "int32":
        data = data.astype("float32") / 2 ** 31
    elif data.dtype == "int16":
        data = data.astype("float32") / 2 ** 15
    elif data.dtype == "uint8":
        data = data.astype("float32") / 2 ** 7
    elif data.dtype == "float32":
        pass
    else:
        raise ValueError("Unsupported data type: {}".format(data.dtype))

    return data, samplerate


def write_wav(file_path: str, samplerate: int, data: np.ndarray, dtype: str = "int16"):
    if dtype == "int32":
        data = data * 2 ** 31
        data = np.int32(data)
    elif dtype == "int16":
        data = data * 2 ** 15
        data = np.int16(data)
    elif dtype == "uint8":
        data = data * 2 ** 7
        data = np.int8(data)
    elif dtype == "float32":
        pass
    else:
        pass
    
    sf.write(file_path, data, samplerate)


class WavEncoder(object):
    def __init__(self, file_path) -> None:
        self.data, self.samplerate = read_wav(file_path)
        if len(self.data.shape) == 1:
            self.data = np.expand_dims(self.data, axis=0)
        self.data = self.data.T  # compatible soundfile format

        self.counter = 0

    def __enter__(self):
        print("enter...")
        return self

    def get(self, block_size: int):
        data_size = self.data.shape[0]
        stereo = self.data.shape[1]
        if data_size % block_size == 0:
            output = self.data
        else:
            # Chunk the data
            output = np.zeros(
                shape=((data_size // block_size) * block_size, stereo),
                dtype=self.data.dtype, # should match with input data
            )
            output[:data_size, :] = self.data[:(data_size // block_size) * block_size, :].copy()

        return output

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exit...")


if __name__ == "__main__":
    # Setup
    blocksize = 64
    file_path = ""

    # Load data
    encoder = WavEncoder(file_path)
    in_data = encoder.get(blocksize)

    # Read each frame
    out_data = np.zeros_like(in_data)
    for iframe in range(in_data.shape[0] // blocksize):
        curr_frame = in_data[iframe * blocksize : (iframe + 1) * blocksize, :]
        out_data[iframe * blocksize : (iframe + 1) * blocksize, :] = curr_frame

    out_file_path = os.getcwd() + "/audio_in_mp3.wav"
    # write_wav(out_file_path, 44100, out_data)
