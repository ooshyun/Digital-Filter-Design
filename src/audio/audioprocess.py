import numpy as np


class ProcessUnit(object):
    """window apply based on overlapping proportion
    """

    overlap = 0.75  # 75% preferred overlap

    def __init__(
        self,
        sampling_freq,
        framesize,
        zeropad=False,
        process_encode=None,
        process_fft=None,
        process_decode=None,
        *args,
        **kwargs
    ):
        self.sampling_freq = sampling_freq
        self.framesize = framesize
        self.window_len = int(framesize / (1 - self.overlap))
        self.zeropad = bool(zeropad)

        if self.zeropad:
            self.nfft = self.window_len * 2
        else:
            self.nfft = self.window_len

        self.pre_fft_buffer = np.zeros(self.window_len)
        self.window_func = np.hanning(self.window_len)
        self.windowed_pre_fft = np.zeros(self.window_len)
        self.fft_data = np.zeros(self.nfft, dtype=complex)
        self.ifft_data = np.zeros(self.nfft)
        self.windowed_ifft = np.zeros(self.window_len)
        self.post_fft_buffer = np.zeros(self.window_len)
        windowfunc_factor = np.mean(self.window_func ** 2)
        self.scale_factor = (1 - self.overlap) / windowfunc_factor

        self.process_encode = process_encode
        self.process_fft = process_fft
        self.process_decode = process_decode

    def stream_process(self, indata):
        channels = indata.shape[1]

        # Reduce to 1-D time series, the method is average
        mixed_data = np.mean(indata, axis=1)  # 1-D format
        
        # signal processing
        processed_data = self._stream_process(mixed_data)  # 1-D format

        # restore the data based on the number of channel
        processed_data = np.tile(processed_data, (channels, 1))

        return processed_data.T

    def _stream_process(self, indata: np.ndarray):
        # time domain process
        if self.process_encode is None:
            processed_time_data = indata.copy()
        else:
            processed_time_data = self.process_encode(indata.copy())

        # overlapping
        self.pre_fft_buffer[:] = np.concatenate(
            (self.pre_fft_buffer[self.framesize :], processed_time_data)
        )

        # windowing
        self.windowed_pre_fft[:] = self.pre_fft_buffer * self.window_func

        # FFT, normalized
        self.fft_data[:] = np.fft.fft(self.windowed_pre_fft, self.nfft)

        # fft domain process
        if self.process_fft is None:
            processed_fft_data = self.fft_data.copy()
        else:
            processed_fft_data = self.process_fft(self.fft_data.copy())

        # iFFT based on window size
        self.ifft_data[:] = np.real(np.fft.ifft(processed_fft_data))

        # apply window, scale factor
        self.windowed_ifft[:] = self.ifft_data[: self.window_len] * self.window_func 

        # overlapping
        self.post_fft_buffer[:] = np.concatenate(
            (self.post_fft_buffer[self.framesize :], np.zeros(self.framesize))
        )
        self.post_fft_buffer[:] = np.add(self.windowed_ifft, self.post_fft_buffer)

        return self.post_fft_buffer[: self.framesize] * self.scale_factor

