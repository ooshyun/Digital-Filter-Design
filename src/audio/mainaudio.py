import numpy as np
import sys

from .audiobuffer import AudioBuffer
from .audioprocess import ProcessUnit
from .audioencoder import WavEncoder, write_wav


def _print_progress(iteration, total, prefix="", suffix="", decimals=1, barLength=50):
    """Represent the progress state through percentage on the terminal.
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = "#" * filledLength + "-" * (barLength - filledLength)
    sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percent, "%", suffix)),
    if iteration == total:
        sys.stdout.write("\n")
    sys.stdout.flush()


class AudioProcess(object):
    """Main audio process
        This class interacts with other applications, audio process unit, and audio buffer.
        
        Parameters
        ----------
        sample_rate (int): The sample rate of the audio data
        frame_size (int): The size of the frame when processing the data
        channels (int): The number of channels of the audio data
        zero_pad (bool): Whether to zero pad the audio data
        audiobuffer (AudioBuffer): The audio buffer
        processunit (ProcessUnit): The process unit
        type_stream (str): The type of stream to process('wav' or '[TODO] mic')
        in_file_path (str): The file path of the input file

        Notes
        -----
        This contains ONLY wav file processing. It will contains [TODO] Mic stream, real-time wave file stream, 
        recording the processeed mic stream. Then, the type_stream can be changed to 'wav' or 'mic', which is currently
        only supported to 'wav'.

        TODO LIST
        ---------
        - Mic stream
        - Wave file stream
        - Recording the processed mic stream
    """

    def __init__(
        self,
        sample_rate,
        frame_size,
        channels,
        zero_pad=False,
        process_time_domain=None,
        process_fft_domain=None,
        process_post_domain=None,
        in_file_path="",
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.channels = channels
        self.zero_pad = zero_pad

        self.audiobuffer = AudioBuffer(frame_size)
        self.processunit = ProcessUnit(
            sample_rate=sample_rate,
            frame_size=frame_size,
            zero_pad=zero_pad,
            process_time_domain=process_time_domain,
            process_fft_domain=process_fft_domain,
            process_post_domain=process_post_domain,
        )

        self.type_stream = "wav"
        self.in_file_path = in_file_path

    def process(self, indata, outdata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)

        # Process the input data
        outdata[:] = self.processunit.stream_process(indata.copy())

        self.audiobuffer.inbuffer = indata
        self.audiobuffer.outbuffer = outdata

    def stream(self):
        if self.type_stream == "wav":
            NotImplementedError
        elif self.type_stream == "mic":
            NotImplementedError
        else:
            raise ValueError("type_stream must be 'wav' or 'mic'")

    def save(self, out_file_path="", progress_bar=True):
        if self.in_file_path == "":
            raise ValueError("in_file_path must be set")

        if self.type_stream == "wav":
            sample_rate = self.sample_rate
            callback = self.process
            frame_size = self.frame_size

            encoder = WavEncoder(self.in_file_path)
            in_data = encoder.get(frame_size)
            del encoder

            out_data = np.zeros_like(in_data)
            num_frames = in_data.shape[0] // frame_size

            for iframe in range(num_frames):
                in_frame = in_data[iframe * frame_size : (iframe + 1) * frame_size, :]
                out_frame = out_data[iframe * frame_size : (iframe + 1) * frame_size, :]
                callback(in_frame, out_frame, frames=None, time=None, status=None)
                out_data[iframe * frame_size : (iframe + 1) * frame_size, :] = out_frame

                if progress_bar:
                    _print_progress(iframe, num_frames)

            if out_file_path == "":
                return out_data
            else:
                write_wav(out_file_path, sample_rate, out_data)

        elif self.type_stream == "mic":
            NotImplementedError
        else:
            raise ValueError("type_stream must be 'wav' or 'mic'")
