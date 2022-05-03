import numpy as np


class ProcessUnit(object):
    """Digital Signal Processing for audio and speech signal

    This class can apply to any sequencial data stream, especially for audio and speech signal.
    Process is based on overlap and window.

	Parameters
	----------
        overlap (float): The overlap ratio between two frames
        sample_rate (int): The sample rate of the signal
        frame_size (int): The frame size of the signal
        zero_pad (bool): Whether to zero-pad the signal
        process_time_domain (function): The function to process the time domain signal
        process_fft_domain (function): The function to process the fft domain signal
        process_post_domain (function): The function to process the post fft domain signal

    Notes
    -----
    This can get the callback function to process function based on its data's domain such as time, frequency and post
    The procedure is as follows:
        1. Set parameters
        2. Get an input frame
        3. Get the number of channel based on its shape
        4. When the channel is over two, Mix the channels
            - Current implementation is to compute average between the channels
        5. Process the frame
            time domain process -> overlap -> window
            -> fft -> fft process -> ifft 
            -> window -> overlap -> [TODO]post process
            -> scale factor -> output
        6. restore the number of channels
        7. return a frame

    TODO LIST
    ---------
    - Add the post process function
    """

    def __init__(
        self,
        sample_rate: int,
        frame_size: int,
        zero_pad=False,
        process_time_domain=None,
        process_fft_domain=None,
        process_post_domain=None,
        *args,
        **kwargs
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.overlap = 0.75
        self.window_len = int(frame_size / (1 - self.overlap))
        self.zero_pad = bool(zero_pad)

        if self.zero_pad:
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

        self.process_time_domain = process_time_domain
        self.process_fft_domain = process_fft_domain
        self.process_post_domain = process_post_domain

    def stream_process(self, indata):
        channels = indata.shape[1]

        # Reduce to 1-D time series, the method is average
        mixed_data = np.mean(indata, axis=1)

        # signal processing
        processed_data = self._wola(mixed_data)

        # restore the data based on the number of channel
        processed_data = np.tile(processed_data, (channels, 1))

        return processed_data.T

    def _wola(self, indata: np.ndarray):
        # time domain process
        if self.process_time_domain is None:
            processed_time_data = indata.copy()
        else:
            processed_time_data = self.process_time_domain(indata.copy())

        # overlapping
        self.pre_fft_buffer[:] = np.concatenate(
            (self.pre_fft_buffer[self.frame_size :], processed_time_data)
        )

        # synthesis window
        self.windowed_pre_fft[:] = self.pre_fft_buffer * self.window_func

        # FFT, normalized
        self.fft_data[:] = np.fft.fft(self.windowed_pre_fft, self.nfft)

        # fft domain process
        if self.process_fft_domain is None:
            processed_fft_data = self.fft_data.copy()
        else:
            processed_fft_data = self.process_fft_domain(self.fft_data.copy())

        # iFFT based on window size
        self.ifft_data[:] = np.real(np.fft.ifft(processed_fft_data))

        # apply synthesis window, scale factor
        self.windowed_ifft[:] = self.ifft_data[: self.window_len] * self.window_func

        # overlapping
        self.post_fft_buffer[:] = np.concatenate(
            (self.post_fft_buffer[self.frame_size :], np.zeros(self.frame_size))
        )
        self.post_fft_buffer[:] = np.add(self.windowed_ifft, self.post_fft_buffer)

        return self.post_fft_buffer[: self.frame_size] * self.scale_factor
