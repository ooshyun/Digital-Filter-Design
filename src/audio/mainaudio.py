import numpy as np
import sys

from .audiobuffer import AudioBuffer
from .audioprocess import ProcessUnit
from .audioencoder import WavEncoder, write_wav

def _print_progress(iteration, total, prefix="", suffix="", decimals=1, barLength=50):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = "#" * filledLength + "-" * (barLength - filledLength)
    sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percent, "%", suffix)),
    if iteration == total:
        sys.stdout.write("\n")
    sys.stdout.flush()

class AudioProcess(object):
    def __init__(
        self,
        sampling_freq,
        framesize,
        channels,
        zeropad=False,
        process_encode=None,
        process_fft=None,
        process_decode=None,
        in_file_path="",
    ) -> None:
        self.sampling_freq = sampling_freq
        self.framesize = framesize
        self.channels = channels
        self.zeropad = zeropad

        self.audiobuffer = AudioBuffer(self.framesize)
        self.processunit = ProcessUnit(
            sampling_freq=self.sampling_freq,
            framesize=self.framesize,
            zeropad=self.zeropad,
            process_encode=process_encode,
            process_fft=process_fft,
            process_decode=process_decode,
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
            raise ValueError("in_file_path and out_file_path must be set")

        if self.type_stream == "wav":
            sampling_freq = self.sampling_freq
            callback = self.process
            frame_size = self.framesize

            encoder = WavEncoder(self.in_file_path)
            in_data = encoder.get(frame_size)
            out_data = np.zeros_like(in_data)
            num_frames = in_data.shape[0] // frame_size

            for iframe in range(num_frames):
                in_frame = in_data[iframe * frame_size : (iframe + 1) * frame_size, :]
                out_frame = out_data[iframe * frame_size : (iframe + 1) * frame_size, :]
                callback(in_frame, out_frame, frames=None, time=None, status=None)
                out_data[iframe * frame_size : (iframe + 1) * frame_size, :] = out_frame

                if progress_bar: _print_progress(iframe, num_frames)

            if out_file_path == "":
                return out_data
            else:
                write_wav(out_file_path, sampling_freq, out_data)

        elif self.type_stream == "mic":
            NotImplementedError
        else:
            raise ValueError("type_stream must be 'wav' or 'mic'")
