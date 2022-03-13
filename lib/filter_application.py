"""Applying Filters to wav file

    This is for pre-fft / fft / post-fft processing wav file using filter.
    It provides the parallel and cascades structure. The details of testing is on
    test.py.
    ---
    TODO LIST 
    [ ] WaveProcessor.process_time_domain
        [ ] 1. Parellel processing
            - How to apply real component and bias ?
            - issue
                0. Matrix M is not include imaginary, the paper said it is.
                1. If applying bias, then under 10 Hz frequency is amplified 10 dB 
                    -> [Trial] adjust the target bias gain to 0
                    -> [Solution] use memory of previous frame's delay
                2. Big peak and big drop make the filter unstable in gain case 1
        [ ] 2. Floating point -> 2.30
        [ ] 3. Double pole position verification -> result is bad
    [ ] sampling frequency return to wav's component to user
    
    - Reference
    https://dsp.stackexchange.com/questions/20194/concept-of-combining-multiple-fir-filters-into-1-fir-filter
    https://kr.mathworks.com/help/audio/ref/graphiceq.html
    https://kr.mathworks.com/help/audio/ug/GraphicEQModel.html
    iCloud/Papers/Audio/Efficient Multi-Band Digital Audio Graphic Equalizer with Accurate Frequency Response Control.pdf

"""
import time
import numpy as np
from scipy.signal import lfilter, hilbert

import scipy.io.wavfile as wav

from lib.graphic_equalizer import EPS
# import matplotlib.pyplot as plt
# from numpy.polynomial.polynomial import polyval as npp_polyval

if __name__.split('.')[0] == 'lib':
    from lib.realtimedsp import wave_file_process, packet
    from lib.config import *
else:
    from realtimedsp import wave_file_process, packet
    from config import *
    
if DEBUG:
    if __name__.split('.')[0] == 'lib':
        from lib.debug.log import PRINTER
    else:
        from debug.log import PRINTER

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
        self._filters = []
        self._freqfilters = []
        self.zi = np.zeros(shape=(31, 2))

        self.timefilter_time = []
        self.freqfilter_time = []
        self.filter_time_olive = []

        # for testing
        self.frame_prev = []
        self.output_prev = []
        self.frame_counter = 0

        self.graphical_equalizer = False

    @property
    def filters(self) -> list:
        return self._filters

    @filters.setter
    def filters(self, coeff):
        self._filters.append(np.array(coeff, dtype=np.float64))

    @property
    def bias(self) -> np.ndarray:
        return self._bias
    
    @bias.setter
    def bias(self, coeff):
        self._bias = np.array(coeff, dtype=np.float64)

    @property
    def freqfilters(self) -> list:
        return self._freqfilters

    @freqfilters.setter
    def freqfilters(self, index):
        self._freqfilters.append(np.array(index).flatten())

    def process_time_domain(self, inpacket : packet, queue, parameter):
        curr_time = time.perf_counter()
        if len(self._filters) == 0:
            return inpacket
        else:
            in_timecount, indata = inpacket.__get__()
            outdata = indata.copy()
            if self.graphical_equalizer:
                result = np.zeros_like(outdata, dtype=np.float)

                coeff = self._filters[0]
                coeff = coeff.reshape(coeff.shape[0], 2, coeff.shape[1] // 2)
                bias = self._bias

                self.zi = self.zi.astype(bias.dtype)

                for i, sample in enumerate(outdata):
                    ptr_zi, ptr_a, ptr_b, b, a = 0, 0, 0, 0, 1
                    
                    sample = sample.astype(bias.dtype)

                    # print(ptr_zi, ptr_a, ptr_b)
                    # first coefficient
                    y = self.zi[:, ptr_zi] + coeff[:, b, ptr_b] * sample
                    # print(self.zi[:, ptr_zi], coeff[:, b, ptr_b],  sample)
                    ptr_a += 1
                    ptr_b += 1

                    # print(ptr_zi, ptr_a, ptr_b)
                    # middle coefficient
                    for _ in range(0, self.zi.shape[-1] - 2 + 1):
                        self.zi[:, ptr_zi] = sample * coeff[:, b, ptr_b] \
                                            - y * coeff[:, a, ptr_a] \
                                            + self.zi[:, ptr_zi+1]
                        ptr_a += 1
                        ptr_b += 1
                        ptr_zi += 1

                    # print(ptr_zi, ptr_a, ptr_b)
                    # last coefficient
                    self.zi[:,
                            ptr_zi] = sample * coeff[:, b, ptr_b] \
                                    - y * coeff[:, a, ptr_a]
                    # update to result
                    result[i] = np.sum(y) + bias*sample

                assert len(outdata) == len(result)
                outdata = result

                if DEBUG:
                    raise ValueError
            else:
                for f in self._filters:
                    b, a = f
                    outdata = lfilter(b, a, outdata)

            self.timefilter_time.append(time.perf_counter() - curr_time)
            outpacket = packet()
            outpacket.timecounter = in_timecount
            outpacket.data = outdata
            return outpacket

    def process_freq_domain(self, inpacket, queue, parameter):
        curr_time = time.perf_counter()
        if len(self._freqfilters) == 0:
            return inpacket
        else:
            _, indata = inpacket.__get__()
            outdata = indata.copy()
            # TODO: Need to fixed for flexibility
            for idx, coeff in enumerate(self._freqfilters):
                outdata[idx + 1 + 56] = outdata[idx + 1 + 56] * coeff
            self.freqfilter_time.append(time.perf_counter() - curr_time)

            outpacket = packet()
            outpacket.timecounter = 0
            outpacket.data = outdata
            return outpacket

    def run(self, savefile_path) -> None:
        wave_file_process(in_file_name=self.wavfile_path,
                          out_file_name=savefile_path,
                          progress_bar=True,
                          stereo=False,
                          overlap=75,
                          block_size=256,
                          zero_pad=False,
                          pre_proc_func=self.process_time_domain,
                          freq_proc_func=self.process_freq_domain)

    def process_time_domain_olivex(self, inpacket):
        """For comparing with olivex
            Plan to delete after applying improved function
            """
        curr_time = time.perf_counter()
        if len(self._filters) == 0:
            return inpacket
        else:
            _, indata = inpacket.__get__()
            outdata = indata.copy()
            for f in self._filters:
                array = outdata
                b, a = f
                # Holder of the previous input frame (X(n))
                previous_input_frame = self.frame_prev
                # Holder of the previous output frame (Y(n))
                previous_output_frame = self.output_prev
                # The frame counter
                frame_counter = self.frame_counter
                # Output array
                output = []
                # Check if the input array is bigger than the filter order
                if (len(array) < len(b)) or (len(array) < len(a)):
                    return None
                # Check if previous frame is empty
                if frame_counter == 0:
                    previous_input_frame = [0] * len(array)
                    previous_output_frame = [0] * len(array)
                # Samples loop
                for n in range(len(array)):
                    # FF and FB values holder
                    Feedback = 0
                    Feedforward = 0
                    # Calculating the Feedback values
                    for i in range(1, len(a)):
                        # Check if the previous output frame data is used
                        if n - i < 0:
                            Feedback = Feedback - (
                                a[i] * previous_output_frame[-i + n])
                        # If n>len(a)-1 then only current output frame data is used
                        else:
                            Feedback = Feedback - (a[i] * output[n - i])
                    # Calculating the Feedforward values
                    for i in range(len(b)):
                        # Check if the previous input frame data is used
                        if n - i < 0:
                            Feedforward = Feedforward + (
                                b[i] * previous_input_frame[-i + n])
                        # If n>len(b)-1 then only current input frame data is used
                        else:
                            Feedforward = Feedforward + (b[i] * array[n - i])
                    output.append((Feedforward + Feedback) / a[0])

                self.frame_prev = array
                self.output_prev = output
                self.frame_counter = frame_counter + 1

            self.filter_time_olive.append(time.perf_counter() - curr_time)

            outpacket = packet()
            outpacket.timecounter = 0
            outpacket.data = outdata
            return outpacket


if __name__ == '__main__':
    import os
    from filt import CustomFilter
    from filter_analyze import FilterAnalyzePlot
    data_path = ''
    result_path = ''

    infile_path = os.path.join(data_path, 'White Noise.wav')
    fs, data = wav.read(infile_path)

    fft_size = 256
    fft_band = np.arange(1, fft_size / 2 + 1) * fs / fft_size
    # fc_band = np.arange(30, 22060, 10)
    fc_band = np.array([100, 1000, 2000, 3000, 5000])

    ploter = FilterAnalyzePlot(sampleing_freq=fs)
    wave_processor = WaveProcessor(wavfile_path=infile_path)

    outfile_path = ''
    outresult_path = ''

    def plot_several_types_filters():
        """Plot the several filters
        """
        fc = 1000
        filter_custom = CustomFilter(sampling_freq=fs,
                                     cutoff_freq=fc,
                                     Qfactor=1 / np.sqrt(2),
                                     gain=3)

        name = 'Shelf Filter'
        # outplot_path = '/Users/seunghyunoh/workplace/document/oliveunion/주간보고_내부/filter/2_process_filter_application_py/image/filter_time_domain_type_'+str(name)+'.jpg'

        # lowpass_filter = filter_custom.lowpass()
        # highpass_filter = filter_custom.highpass()
        # bandpass_filter = filter_custom.bandpass()
        # notch_filter = filter_custom.notch()
        # peak_filter = filter_custom.peaking()
        shelf_filter = filter_custom.shelf()

        ploter.filters = shelf_filter

        ploter.plot(save_path=None, name=name)

    def wav_process():
        """Comparison between time domain and frequency domain
        """
        fc = 1033.59375
        # time
        filter_custom = CustomFilter(sampling_freq=fs,
                                     cutoff_freq=fc,
                                     Qfactor=10,
                                     gain=3)
        peak_filter = filter_custom.peaking()
        wave_processor.filters = peak_filter
        ploter.filters = peak_filter

        # frequency
        idfreq = np.argwhere(fft_band == fc)
        wave_processor.freqfilters = idfreq

        # process
        outfile_path = '2_process_filter_application_py/test.wav'
        outresult_path = os.path.join(result_path, outfile_path)
        wave_processor.run(savefile_path=outresult_path)

        print(
            sum(wave_processor.freqfilter_time) /
            len(wave_processor.freqfilter_time))
        print(
            sum(wave_processor.timefilter_time) /
            len(wave_processor.timefilter_time))
