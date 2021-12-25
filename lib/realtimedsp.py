"""Real time DSP process Libraray
    This library provides an easy solution to apply new algorithms with
    no need to stress with the frame overlap for that.

    There are different functions depending in the data input:

    -----------------------------------------------------------------------------------
    wave_file_process
    Function reads a wave file and call the processing functions entered by the user
    User can return the process data or save it in an wav file
        * Params:
        - in_file_name: the wave file name (with full directory)
        - get_file_details: returns the wav file details in form of dictionary with keys
                            name, samplerate, channels, length_samples, length_seconds and
                            data_type
        - out_file_name: the output wav file name. If empty will return processed data in
                        python list format.
        - stereo: True for stereo, False for mono (left will be used for mono). If the
                  input file is mono, left data will be copied to right.
        - overlap: 50% (50) or 75% (75) overlap options
        - block_size: The number of input samples per frame
        - zero_pad: Add nff/2 zeros in each frame, reducing by half the input frame
        of the system. For example, if overlap is 50% and NFFT is 256 point, each
        time domain frame will have 64 new samples per loop if zero pad is True
        and 128 if zero pad is False.
        - pre_proc_func: The time domain preprocess function
            * Inputs: The current time domain frames
            * Output: Should return the same object as the input with processed data
            * Format: Python list of float objects
                      ([1,2,3...] for mono [[1,2,3..][1,2,3..]] for stereo)
        - freq_proc_func: The frequency domain process function
            * Inputs: The current frequency domain frames with real and imaginary
                    components of size (nfft/2)+1 (non normalized)
            * Output: Should return the same object as the input with processed data
            * Format: Python list of numpy.complex128 objects
                      ([1,2,3...] for mono [[1,2,3..][1,2,3..]] for stereo)
        - post_proc_func: The time domain postprocess function
            * Inputs: The current time domain frames
            * Output: Should return the same object as the input with processed data
            * Format: Python list of float objects
                      ([1,2,3...] for mono [[1,2,3..][1,2,3..]] for stereo)
    ---
    Update 2021-11-10
    - Added the time packet class to share the parameter for processing
        - Data in each frame
        - Time stamp
        - Parameter
        - Queue
    ---
    TODO LIST
        [ ] 1. Refactor the code to make it more abstract
        [ ] 2. Processing for a frame
        [ ] 3. Processing wav using processing for a frame
        [ ] 4. make class packet to be simple
    -----------------------------------------------------------------------------------
"""
import math
import numpy as np
import scipy.io.wavfile as wav
import sys


class packet(object):

    def __init__(self):
        self.timecounter = None
        self.dataLength = None
        self.data = None

    def __get__(self):
        if self.timecounter is None or self.data is None:
            return None
        else:
            return self.timecounter, self.data

    def set(self, timecounter=None, data=None):
        if timecounter is None:
            self.data = data
            self.dataLength = len(self.data)
        elif data is None:
            self.timecounter = timecounter
        else:
            self.timecounter = timecounter
            self.data = data
            self.dataLength = len(self.data)

    def copy(self):
        buf_packet = packet()
        buf_packet.set(timecounter=self.timecounter, data=self.data)
        return buf_packet


def _print_progress(iteration,
                    total,
                    prefix='',
                    suffix='',
                    decimals=1,
                    barLength=50):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def _get_file_details(in_file_name):
    wav_freq, wav_data = wav.read(in_file_name)
    outdata = {"name": in_file_name, "samplerate": wav_freq}
    try:
        outdata["channels"] = wav_data.shape[1]
        outdata["length_samples"] = wav_data.shape[0]
        outdata["length_seconds"] = wav_data.shape[0] / wav_freq
    except:
        outdata["channels"] = 1
        outdata["length_samples"] = len(wav_data)
        outdata["length_seconds"] = len(wav_data) / wav_freq
    if wav_data.dtype == 'int16':
        outdata["data_type"] = 'int16'
    elif wav_data.dtype == 'int32':
        outdata["data_type"] = 'int32'
    elif wav_data.dtype == 'float32':
        outdata["data_type"] = 'float32'

    return outdata


def wave_file_process(in_file_name="",
                      get_file_details=False,
                      out_file_name="",
                      progress_bar=True,
                      stereo=True,
                      overlap=75,
                      block_size=128,
                      zero_pad=True,
                      pre_proc_func=None,
                      freq_proc_func=None,
                      post_proc_func=None,
                      queue_shared=None,
                      parameter=None,
                      full_freq_proc_func=None):
    if not get_file_details:
        print("\n#################  Wave File Process #################")

    if isinstance(in_file_name, str):
        # Errors check before do any process
        if in_file_name == "":
            print("ERROR: No wav file name entered")
            return None
        # Reading the wave file
        wav_freq, wav_data = wav.read(in_file_name)
    elif isinstance(in_file_name, dict):
        wav_freq, wav_data = in_file_name['sr'], in_file_name['data']
    else:
        print("ERROR: No wav file name entered")
        return None

    print("Processing the " + (in_file_name.split("/")[-1]) + " file...")
    # Returns the details from file name entered
    if get_file_details:
        return _get_file_details(in_file_name)

    if (pre_proc_func is None) and (freq_proc_func is None) and (post_proc_func
                                                                 is None):
        print(
            "WARNING: No process function entered, no process will be applied to the file"
        )
        func_bool = input("Do you want to continue (Y/n) ? ")
        if func_bool != "Y" and func_bool != "y":
            return None

    if overlap != 50 and overlap != 75:
        print("ERROR: Wrong overlap value (50 or 75 only)")
        return None

    if (block_size % 2) != 0:
        print("ERROR: Blocksize value should be even number")
        return None

    # Checking the input bit depth (no support for 8 and 24 bit PCM)
    bit_depth = 0
    if wav_data.dtype == 'int16':
        bit_depth = 15
    elif wav_data.dtype == 'int32':
        bit_depth = 31
    elif wav_data.dtype == 'float32':
        bit_depth = 0
    elif wav_data.dtype == 'float64':
        bit_depth = 0

    # Checking the NFFT value
    nfft = 0
    if zero_pad:
        if overlap == 75:
            nfft = int(block_size * 8)
        else:
            nfft = int(block_size * 4)
    else:
        if overlap == 75:
            nfft = int(block_size * 4)
        else:
            nfft = int(block_size * 2)

    # Checking the FFT frame size
    if zero_pad:
        fft_frame_size = int(nfft / 2)
    else:
        fft_frame_size = nfft

    # Checking if the overlap is 50% or 75%
    new_frame_size = 0
    if overlap == 50:
        new_frame_size = int(fft_frame_size / 2)
    elif overlap == 75:
        new_frame_size = int(fft_frame_size / 4)

    # Getting the window function
    hanning = np.array([(0.5 - (0.5 * math.cos(
        (2 * math.pi * i) / (fft_frame_size - 1))))
                        for i in range(fft_frame_size)],
                       dtype='float64')
    """Initializing the static variables used in WOLA operation"""
    output_index_start, output_index_end = 0, new_frame_size
    # Checking if input file is stereo and normalizing buffers

    wav_data_left = None
    wav_data_right = None
    if len(wav_data.shape) == 1:
        wav_data_left = wav_data / (2**bit_depth)
        if stereo:
            wav_data_right = wav_data / (2**bit_depth)

    elif len(wav_data.shape) == 2:
        wav_data_left = ((wav_data[:, 0] / (2**bit_depth)) +
                         (wav_data[:, 1] / (2**bit_depth))) / 2
        if stereo:
            wav_data_right = wav_data[:, 1] / (2**bit_depth)
    else:
        raise ValueError("ERROR: Wrong number of dimensions in wav file")

    # Getting the different frames for left and right
    frames_data_left = [
        wav_data_left[(i * new_frame_size):(i * new_frame_size +
                                            new_frame_size)]
        for i in range(int(len(wav_data_left) / new_frame_size))
    ]

    # The final output list with the entire data processed
    output_left = np.zeros(len(wav_data))

    # Previous loop windowed frame used for WOLA operation
    windowed_frame_left = np.zeros(fft_frame_size - new_frame_size)

    # Input buffer for WOLA operation
    input_frames_left = np.zeros(fft_frame_size)

    # FFT input buffer
    fft_in_left = np.zeros(nfft)

    # First half of the FFT frequency data
    fft_channels_left = np.zeros(int(nfft / 2) + 1)

    # IFFT output with only real data format
    ifft_out_left = np.zeros(fft_frame_size)

    # The output time domain frame before post process applied
    out_frame_left = np.zeros(new_frame_size)

    if stereo:
        frames_data_right = [
            wav_data_right[(i * new_frame_size) \
                            :(i * new_frame_size + new_frame_size)]
            for i in range(int(len(wav_data_left) / new_frame_size))
        ]
        output_right = np.zeros(len(wav_data))
        windowed_frame_right = np.zeros(fft_frame_size - new_frame_size)
        input_frames_right = np.zeros(fft_frame_size)
        fft_in_right = np.zeros(nfft)
        fft_channels_right = np.zeros(int(nfft / 2) + 1)
        ifft_out_right = np.zeros(fft_frame_size)
        out_frame_right = np.zeros(new_frame_size)

    counter = packet()

    # The data frames loop
    for index in range(len(frames_data_left)):
        counter = packet()
        counter.set(timecounter=index)
        # tic = tm.perf_counter()
        """Initialize Variable"""
        new_frame_left, new_frame_right = None, None
        pre_processed_frame_left, pre_processed_frame_right = None, None
        fft_out_left, fft_out_right = None, None
        full_freq_processed_frame_left, full_freq_processed_frame_right = None, None
        freq_processed_frame_left, freq_processed_frame_right = None, None
        ifft_in_left, ifft_in_right = None, None
        post_processed_frame_left, post_processed_frame_right = None, None
        """Loading the current unprocessed time domain frame"""
        new_frame_left = frames_data_left[index]
        if stereo:
            new_frame_right = frames_data_right[index]
        """Preprocess Function"""
        # Checking if No preprocess function just copies the input
        if not pre_proc_func:
            pre_processed_frame_left = new_frame_left
            if stereo:
                pre_processed_frame_right = new_frame_right

        # Calling the preprocess function with the new frame input
        else:
            if stereo:
                # Adding the 2 streams in one list
                indata = [new_frame_left, new_frame_right]
                counter.set(data=indata)

                _, outdata = pre_proc_func(counter, queue_shared,
                                           parameter).__get__()

                pre_processed_frame_left = outdata[0]
                pre_processed_frame_right = outdata[1]
            else:
                indata = new_frame_left
                counter.set(data=indata)

                _, pre_processed_frame_left = pre_proc_func(
                    counter, queue_shared, parameter).__get__()
        """Overlap"""
        # Checking if 50% overlap to set the FFT input buffer
        if overlap == 50:
            input_frames_left[:block_size] = input_frames_left[block_size:]
            input_frames_left[block_size:] = pre_processed_frame_left
            if stereo:
                input_frames_right[:block_size] = input_frames_right[
                    block_size:]
                input_frames_right[block_size:] = pre_processed_frame_right

        # Checking if 75% overlap to set the FFT input buffer
        elif overlap == 75:
            input_frames_left[:3 *
                              block_size] = input_frames_left[block_size:4 *
                                                              block_size]
            input_frames_left[3 * block_size:4 *
                              block_size] = pre_processed_frame_left
            if stereo:
                input_frames_right[:3 * block_size] = input_frames_right[
                    block_size:4 * block_size]
                input_frames_right[3 * block_size:4 *
                                   block_size] = pre_processed_frame_right

        else:
            raise ValueError("Overlap must be 50% or 75%")

        # Applying the window function (Hanning)
        fft_in_left[:fft_frame_size] = input_frames_left * hanning
        if stereo:
            fft_in_right[:fft_frame_size] = input_frames_right * hanning
        """FFT operation"""
        fft_out_left = np.fft.fft(fft_in_left)
        if stereo:
            fft_out_right = np.fft.fft(fft_in_right)

        # No frequency domain process function using full frame just copies the input
        if not full_freq_proc_func:
            full_freq_processed_frame_left = fft_out_left
            if stereo:
                full_freq_processed_frame_right = fft_out_right

        # Calling the frequency domain process function
        else:
            if stereo:
                indata = [fft_out_left, fft_out_right]
                counter.set(data=indata)

                _, outdata = full_freq_proc_func(counter, queue_shared,
                                                 parameter).__get__()

                full_freq_processed_frame_left = outdata[0]
                full_freq_processed_frame_right = outdata[1]
            else:
                indata = fft_out_left
                counter.set(data=indata)

                _, full_freq_processed_frame_left = full_freq_proc_func(
                    counter, queue_shared, parameter).__get__()
        """Removing reflection"""
        fft_channels_left = full_freq_processed_frame_left[:(int(nfft / 2) + 1)]
        if stereo:
            fft_channels_right = full_freq_processed_frame_right[:(
                int(nfft / 2) + 1)]

        # Checking if there is a frequency domain process function
        # No frequency domain process function just copies the input
        if not freq_proc_func:
            freq_processed_frame_left = fft_channels_left
            if stereo:
                freq_processed_frame_right = fft_channels_right

        # Calling the frequency domain process function
        else:
            if stereo:
                # Adding the 2 streams in one list
                indata = [fft_channels_left, fft_channels_right]
                counter.set(data=indata)

                _, outdata = freq_proc_func(counter, queue_shared,
                                            parameter).__get__()

                freq_processed_frame_left = outdata[0]
                freq_processed_frame_right = outdata[1]
            else:
                indata = fft_channels_left
                counter.set(data=indata)

                _, freq_processed_frame_left = freq_proc_func(
                    counter, queue_shared, parameter).__get__()
        """Mirror the spectrum"""
        fft_out_left[:int(nfft / 2)] = freq_processed_frame_left[:int(nfft / 2)]

        # Mirror copy except bias
        temp_mirror = freq_processed_frame_left[1:int(nfft / 2)]

        # To make the reverse bin array for mirror
        temp_mirror = temp_mirror[::-1]
        fft_out_left[int(nfft / 2) + 1:] = np.conj(temp_mirror)

        # Nyquist
        fft_out_left[int(nfft / 2)] = freq_processed_frame_left[int(nfft / 2)]
        if stereo:
            fft_out_right[:int(nfft /
                               2)] = freq_processed_frame_right[:int(nfft / 2)]

            # Mirror copy except bias
            temp_mirror = freq_processed_frame_right[1:int(nfft / 2)]
            temp_mirror = temp_mirror[::-1]
            fft_out_right[int(nfft / 2) + 1:] = np.conj(temp_mirror)

            # Nyquist
            fft_out_right[int(nfft / 2)] = freq_processed_frame_right[int(nfft /
                                                                          2)]
        """FFT operation"""
        ifft_in_left = np.fft.ifft(fft_out_left)
        if stereo:
            ifft_in_right = np.fft.ifft(fft_out_right)
        """Overlap"""
        # Separating the desired part of the IFFT output
        # For 50% overlap no second window needed
        if overlap == 50:
            ifft_out_left = ifft_in_left[:fft_frame_size].real
            out_frame_left = ifft_out_left[:new_frame_size] \
                            + windowed_frame_left[:new_frame_size]
            windowed_frame_left = ifft_out_left[new_frame_size:]
            if stereo:
                ifft_out_right = ifft_in_right[:fft_frame_size].real
                out_frame_right = ifft_out_right[:new_frame_size] \
                                + windowed_frame_right[:new_frame_size]
                windowed_frame_right = ifft_out_right[new_frame_size:]

        # For 75% applying second window with gain adjustment to normalize output (-1< out <1)
        elif overlap == 75:
            ifft_out_left = np.float64(
                ifft_in_left[:fft_frame_size].real) * hanning * 2 / 3
            out_frame_left = ifft_out_left[:new_frame_size] \
                            + windowed_frame_left[:new_frame_size]
            windowed_frame_left[:2 * new_frame_size] = windowed_frame_left[new_frame_size:3 * new_frame_size] \
                                                    + ifft_out_left[new_frame_size:3 * new_frame_size]
            windowed_frame_left[2 * new_frame_size:] = ifft_out_left[
                3 * new_frame_size:]
            if stereo:
                ifft_out_right = np.float64(
                    ifft_in_right[:fft_frame_size].real) * hanning * 2 / 3
                out_frame_right = ifft_out_right[:new_frame_size] \
                                + windowed_frame_right[:new_frame_size]
                windowed_frame_right[:2 * new_frame_size] = windowed_frame_right[new_frame_size:3 * new_frame_size] \
                                                        + ifft_out_left[new_frame_size:3 * new_frame_size]
                windowed_frame_right[2 * new_frame_size:] = ifft_out_right[
                    3 * new_frame_size:]
        """Post process"""
        # Checking if there is a post process function
        # No post process function just copies the input
        if not post_proc_func:
            post_processed_frame_left = out_frame_left
            if stereo:
                post_processed_frame_right = out_frame_right

        # Calling the post process function
        else:
            # Adding the 2 streams in one list
            if stereo:
                indata = [
                    np.empty(len(new_frame_left)),
                    np.empty(len(new_frame_left))
                ]
                indata[0] = out_frame_left
                indata[1] = out_frame_right
                counter.set(data=indata)

                _, outdata = post_proc_func(counter, queue_shared).__get__()

                post_processed_frame_left = outdata[0]
                post_processed_frame_right = outdata[1]

            else:
                indata = out_frame_left
                counter.set(data=indata)
                _, post_processed_frame_left = post_proc_func(
                    counter, queue_shared).__get__()
        """Appending to Output"""
        # Adding the current frame to the output buffer
        output_left[
            output_index_start:
            output_index_end] = post_processed_frame_left[:new_frame_size]
        if stereo:
            output_right[
                output_index_start:
                output_index_end] = post_processed_frame_right[:new_frame_size]

        output_index_start += new_frame_size
        output_index_end += new_frame_size
        """Updating the progress bar"""
        if progress_bar:
            _print_progress(index, len(frames_data_left))

        # print(f"Period {tm.perf_counter()-tic}")

    print("\nProcess COMPLETE!")
    # Return the output buffers Left and Right if out_file_name=""
    if (out_file_name == "") or (out_file_name is None):
        if stereo:
            outdata = [output_left, output_right]
            return outdata
        else:
            return output_left

    # Else save the output wav file
    else:
        left = np.float32(output_left)
        data = left
        if stereo:
            right = np.float32(output_right)
            data = np.vstack((left, right)).T
        wav.write(out_file_name, wav_freq, data)


if __name__ == "__main__":
    in_file = '/Users/seunghyunoh/workplace/study/filterdesign/ExampleMusic/White Noise.wav'
    out_file = '/Users/seunghyunoh/workplace/study/filterdesign/ExampleMusic/White Noise Test.wav'
    wave_file_process(in_file_name=in_file,
                      out_file_name=out_file,
                      stereo=False,
                      block_size=512,
                      progress_bar=True)

    in_file = '/Users/seunghyunoh/workplace/study/filterdesign/ExampleMusic/White Noise Stereo.wav'
    out_file = '/Users/seunghyunoh/workplace/study/filterdesign/ExampleMusic/White Noise Stereo Test.wav'
    wave_file_process(in_file_name=in_file,
                      out_file_name=out_file,
                      stereo=True,
                      block_size=512,
                      progress_bar=True)
