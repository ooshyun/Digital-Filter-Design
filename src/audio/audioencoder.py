"""Encoder for wav, pcm dataset including Audio and Speech
"""

import librosa
import soundfile as sf
import numpy as np


def read_wav(file_path: str):
    """Read wave file and return the data and the sampling frequency
    """
    data, sample_rate = librosa.load(file_path, sr=44100, mono=False)
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

    return data, sample_rate


def write_wav(file_path: str, sample_rate: int, data: np.ndarray, dtype: str = "int16"):
    """Write wave file to path
    """
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

    sf.write(file_path, data, sample_rate)


class WavEncoder(object):
    """Encoder for wav, pcm dataset including Audio and Speech
    
        This class read wav file, trasform the data to numpy ndarray for sound device format.
        It chunk the data based on given the size of frame.

        Parameters
        ----------
                file_path (type): the path for input file

    """

    def __init__(self, file_path: str) -> None:
        self.data, self.sample_rate = read_wav(file_path)
        if len(self.data.shape) == 1:
            self.data = np.expand_dims(self.data, axis=0)
        self.data = self.data.T  # compatible soundfile format

    def get(self, frame_size: int):
        data_size = self.data.shape[0]
        stereo = self.data.shape[1]
        if data_size % frame_size == 0:
            output = self.data
        else:
            output = np.zeros(
                shape=((data_size // frame_size) * frame_size, stereo),
                dtype=self.data.dtype,  # should match with input data
            )
            output[:data_size, :] = self.data[
                : (data_size // frame_size) * frame_size, :
            ].copy()

        return output
