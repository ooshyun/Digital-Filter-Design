"""Applying Filters to wav file

    This is for pre-fft / fft / post-fft processing wav file using filter.
    It provides the parallel and cascades structure. The details of testing is on
    test.py.
    ---
    TODO LIST 
    [ ] WaveProcessor.process_time_domain
        [ ] 1. Parellel processing
        [ ] 2. Floating point -> 2.30
    [ ] sampling frequency return to wav's component to user
    
    - Reference
        https://dsp.stackexchange.com/questions/20194/concept-of-combining-multiple-fir-filters-into-1-fir-filter
        https://kr.mathworks.com/help/audio/ref/graphiceq.html
        https://kr.mathworks.com/help/audio/ug/GraphicEQModel.html
"""
import time
import numpy as np
from scipy.signal import lfilter, lfilter_zi, filtfilt
import scipy.io.wavfile as wav

from .audio import AudioProcess
from .config import DEBUG

if DEBUG:
    from .debugging import maker_logger, check_time, print_func_time

    PRINTER = maker_logger()


class WaveProcessor(object):
    """Process wave file using filters
    """

    def __init__(self, wavfile_path) -> None:
        if isinstance(wavfile_path, str):
            self.wavfile_path = wavfile_path
            self.sampleing_freq, self.data_wav = wav.read(wavfile_path)
        elif isinstance(wavfile_path, np.ndarray):
            self.wavfile_path = None
            self.data_wav = wavfile_path
            self.sampleing_freq = 48000
        else:
            raise ValueError("wavfile_path must be str or np.ndarray")

        self._bias = None
        self._filter_time_domain_list = []
        self._filter_freq_domain_list = []
        self.zi = []

        self.time_filter_time = []
        self.time_filter_freq = []

        # for testing
        self.frame_prev = []
        self.output_prev = []
        self.frame_counter = 0

        self.graphical_equalizer = False

    @property
    def filter_time_domain_list(self) -> list:
        return self._filter_time_domain_list

    @filter_time_domain_list.setter
    def filter_time_domain_list(self, coeff):
        if not isinstance(coeff, np.ndarray):
            coeff = np.array(coeff)
        self._filter_time_domain_list.append(np.array(coeff, dtype=np.float64))
        self.zi.append(
            np.zeros(shape=(coeff.shape[0], coeff.shape[-1] // 2 - 1), dtype=np.float64)
        )

    @property
    def bias(self) -> np.ndarray:
        return self._bias

    @bias.setter
    def bias(self, coeff):
        self._bias = np.array(coeff, dtype=np.float64)

    @property
    def filter_freq_domain_list(self) -> list:
        return self._filter_freq_domain_list

    @filter_freq_domain_list.setter
    def filter_freq_domain_list(self, index):
        self._filter_freq_domain_list.append(np.array(index).flatten())

    def process_time_domain(self, indata):
        curr_time = time.perf_counter()
        if len(self._filter_time_domain_list) == 0:
            return indata
        else:
            frame = indata.copy()
            outdata = np.zeros_like(frame)

            if self.graphical_equalizer:
                result = np.zeros_like(outdata, dtype=np.float)

                coeff = self._filter_time_domain_list[0]
                coeff = coeff.reshape(coeff.shape[0], 2, coeff.shape[1] // 2)
                zi = self.zi[0]
                bias = self._bias

                for i, sample in enumerate(outdata):
                    ptr_zi, ptr_a, ptr_b, b, a = 0, 0, 0, 0, 1
                    sample = sample.astype(bias.dtype)

                    # first coefficient
                    y = zi[:, ptr_zi] + coeff[:, b, ptr_b] * sample
                    ptr_a += 1
                    ptr_b += 1

                    # middle coefficient
                    for _ in range(0, zi.shape[-1] - 2 + 1):
                        zi[:, ptr_zi] = (
                            sample * coeff[:, b, ptr_b]
                            - y * coeff[:, a, ptr_a]
                            + zi[:, ptr_zi + 1]
                        )
                        ptr_a += 1
                        ptr_b += 1
                        ptr_zi += 1

                    # last coefficient
                    zi[:, ptr_zi] = sample * coeff[:, b, ptr_b] - y * coeff[:, a, ptr_a]
                    # update to result
                    result[i] = np.sum(y) + sample * bias

                assert len(outdata) == len(result)
                outdata = result
            else:
                for f in self._filter_time_domain_list:
                    b, a = f
                    outdata = lfilter(b, a, frame)
            self.time_filter_time.append(time.perf_counter() - curr_time)
            return outdata

    def process_freq_domain(self, indata):
        curr_time = time.perf_counter()
        if len(self._filter_freq_domain_list) == 0:
            return indata
        else:
            outdata = indata.copy()
            
            for coeff in self._filter_freq_domain_list:
                outdata[:outdata.shape[0]//2+1] = outdata[:outdata.shape[0]//2+1] * coeff
                
            outdata[outdata.shape[0]//2+1:] = np.conjugate(outdata[1:outdata.shape[0]//2])
            self.time_filter_freq.append(time.perf_counter() - curr_time)
            return outdata

    def run(self, savefile_path) -> None:        
        audio_process = AudioProcess(sampling_freq=self.sampleing_freq,
                                    framesize=64,
                                    channels=1,
                                    zeropad=False,
                                    process_encode=self.process_time_domain,
                                    process_fft=self.process_freq_domain,
                                    in_file_path=self.wavfile_path)
                                    
        audio_process.save(out_file_path=savefile_path)
