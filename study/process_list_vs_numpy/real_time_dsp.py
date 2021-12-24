##################################################################################################
#                                                                                                #
#                                 Real time DSP process Lib                                      #
#                                                                                                #
##################################################################################################
"""
    This library provides an easy solution to apply new algorithms with
    no need to stress with the frame overlap for that.

    There are different functions depending in the data input:

    -----------------------------------------------------------------------------------
    # wave_file_process
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
    -----------------------------------------------------------------------------------

"""

########################################## Imports #############################################
import math
import numpy as np
import scipy.io.wavfile as wav
import sys
import time as tm

# Test
from compare import compare_array
offset = 1e-15
flag_debug = 1

######################################### WAV File Process #############################################
def _print_progress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
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
    outdata = {}
    outdata["name"] = in_file_name
    outdata["samplerate"] = wav_freq
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
                      mode=0):
    # if mode == 0:
    #     mode_numpy = 0
    #     mode_list = 1
    # elif mode == 1:
    #     mode_numpy = 1
    #     mode_list = 0

    mode_numpy = 1
    mode_list = 1

    if not get_file_details:
        print("\n#################  Wave File Process #################")

    # Errors check before do any process
    if in_file_name == "":
        if flag_debug:
            import debug
            debug.print_debug()
        print("ERROR: No wav file name entered")
        return None

    # Returns the details from file name entered
    if get_file_details:
        return _get_file_details(in_file_name)

    if (pre_proc_func is None) and (freq_proc_func is None) and (post_proc_func is None):
        print("WARNING: No process function entered, no process will be applied to the file")
        func_bool = "Y"
        # input("Do you want to continue (Y/n) ? ")
        if func_bool != "Y" and func_bool != "y":
            return None

    if overlap != 50 and overlap != 75:
        if flag_debug:
            import debug
            debug.print_debug()
        print("ERROR: Wrong overlap value (50 or 75 only)")
        return None

    if (block_size % 2) != 0:
        if flag_debug:
            import debug
            debug.print_debug()
        print("ERROR: Blocksize value should be even number")
        return None

    print("Processing the " + (in_file_name.split("/")[-1]) + " file...")

    # Reading the wave file
    wav_freq, wav_data = wav.read(in_file_name)
    bit_depth = 0
    # Checking the input bit depth (no support for 8 and 24 bit PCM)
    if wav_data.dtype == 'int16':
        bit_depth = 15
    elif wav_data.dtype == 'int32':
        bit_depth = 31
    elif wav_data.dtype == 'float32':
        bit_depth = 0
    elif wav_data.dtype == 'float64':
        bit_depth = 0

    wav_data_left = None
    wav_data_right = None

    # Checking if input file is stereo and normalizing buffers
    if len(wav_data.shape) == 1:
        wav_data_left = wav_data / (2 ** bit_depth)
        if stereo:
            wav_data_right = wav_data / (2 ** bit_depth)

    elif len(wav_data.shape) == 2:
        if stereo:
            wav_data_left = wav_data[:, 0] / (2 ** bit_depth)
            wav_data_right = wav_data[:, 1] / (2 ** bit_depth)
        else:
            wav_data_left = ((wav_data[:, 0] / (2 ** bit_depth)) + (wav_data[:, 1] / (2 ** bit_depth))) / 2

    # Check the Period
    period_list = []
    result_list = []

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

    # Getting the different frames for left and right
    frames_data_left = [wav_data_left[(i * new_frame_size):(i * new_frame_size + new_frame_size)] for i in
                        range(int(len(wav_data_left) / new_frame_size))]
    frames_data_right = None
    if stereo:
        frames_data_right = [wav_data_right[(i * new_frame_size):(i * new_frame_size + new_frame_size)] for i in
                             range(int(len(wav_data_left) / new_frame_size))]

    # Getting the window function
    hanning = [(0.5 - (0.5 * math.cos((2 * math.pi * i) / (fft_frame_size - 1)))) for i in range(fft_frame_size)]
    if mode_numpy:
        hanning_np = np.array(hanning)

    # Initializing the static variables used in WOLA operation
    # The final output list with the entire data processed
    output_left = None
    output_right = None
    output_left_np = None
    output_right_np = None

    output_index_start = 0
    output_index_end = new_frame_size

    if mode_list:
        output_left = []
        output_right = []

    if mode_numpy:
        output_left_np = np.zeros(len(wav_data))
        output_right_np = np.zeros(len(wav_data))

    # Previous loop windowed frame used for WOLA operation
    windowed_frame_left = None
    windowed_frame_right = None
    windowed_frame_left_np = None
    windowed_frame_right_np = None

    if mode_list:
        windowed_frame_left = [0] * (fft_frame_size - new_frame_size)
        windowed_frame_right = [0] * (fft_frame_size - new_frame_size)

    if mode_numpy:
        windowed_frame_left_np = np.zeros(fft_frame_size - new_frame_size)
        windowed_frame_right_np = np.zeros(fft_frame_size - new_frame_size)

    input_frames_left = None
    input_frames_right = None
    input_frames_left_np = None
    input_frames_right_np = None

    # Input buffer for WOLA operation
    if mode_list:
        input_frames_left = [0] * fft_frame_size
        input_frames_right = [0] * fft_frame_size

    if mode_numpy:
        input_frames_left_np = np.zeros(fft_frame_size)
        input_frames_right_np = np.zeros(fft_frame_size)

    fft_in_left = None
    fft_in_right = None
    fft_in_left_np = None
    fft_in_right_np = None

    # FFT input buffer
    if mode_list:
        fft_in_left = [0] * nfft
        fft_in_right = [0] * nfft

    if mode_numpy:
        fft_in_left_np = np.zeros(nfft)
        fft_in_right_np = np.zeros(nfft)

    # First half of the FFT frequency data
    if mode_list:
        fft_channels_left = [0] * (int(nfft / 2) + 1)
        fft_channels_right = [0] * (int(nfft / 2) + 1)

    if mode_numpy:
        fft_channels_left_np = np.zeros(int(nfft / 2) + 1)
        fft_channels_right_np = np.zeros(int(nfft / 2) + 1)

    # IFFT output with only real data format
    if mode_list:
        ifft_out_left = [0] * fft_frame_size
        ifft_out_right = [0] * fft_frame_size

    if mode_numpy:
        ifft_out_left_np = np.zeros(fft_frame_size)
        ifft_out_right_np = np.zeros(fft_frame_size)
    # The output time domain frame before post process applied

    out_frame_left = None
    out_frame_right = None
    out_frame_left_np = None
    out_frame_right_np = None

    if mode_list:
        out_frame_left = [0] * new_frame_size
        out_frame_right = [0] * new_frame_size

    if mode_numpy:
        out_frame_left_np = np.zeros(new_frame_size)
        out_frame_right_np = np.zeros(new_frame_size)

    # The data frames loop
    for frame in range(len(frames_data_left)):
        tic = tm.perf_counter()
        # Loading the current unprocessed time domain frame

        new_frame_left = None
        new_frame_right = None

        new_frame_left_np = None
        new_frame_right_np = None
        if mode_list:
            new_frame_left = [frames_data_left[frame][i] for i in range(new_frame_size)]
            if stereo:
                new_frame_right = [frames_data_right[frame][i] for i in range(new_frame_size)]

        if mode_numpy:
            new_frame_left_np = frames_data_left[frame][:new_frame_size]
            if stereo:
                new_frame_right_np = frames_data_right[frame][:new_frame_size]

        # Debugging
        # compare_array(new_frame_left, new_frame_left_np, offset)
        # if stereo:
        #     compare_array(new_frame_right, new_frame_right_np, offset)

        # Checking if there is a preprocess function set
        # No preprocess function just copies the input
        pre_processed_frame_left = None
        pre_processed_frame_right = None
        pre_processed_frame_left_np = None
        pre_processed_frame_right_np = None

        if not pre_proc_func:
            if mode_list:
                pre_processed_frame_left = new_frame_left
                if stereo:
                    pre_processed_frame_right = new_frame_right

            if mode_numpy:
                pre_processed_frame_left_np = new_frame_left_np
                if stereo:
                    pre_processed_frame_right_np = new_frame_right_np

        # Calling the preprocess function with the new frame input
        else:
            if mode_list:
                if stereo:
                    # Adding the 2 streams in one list
                    indata = [new_frame_left, new_frame_right]
                    outdata = pre_proc_func(indata)

                    pre_processed_frame_left = outdata[0]
                    pre_processed_frame_right = outdata[1]
                else:
                    pre_processed_frame_left = pre_proc_func(new_frame_left)

            if mode_numpy:
                if stereo:
                    indata_np = np.empty((0, len(new_frame_left_np)))
                    indata_np = np.append(indata_np, [new_frame_left_np], axis=0)
                    indata_np = np.append(indata_np, [new_frame_right_np], axis=0)
                    outdata_np = pre_proc_func(indata_np)

                    pre_processed_frame_left_np = outdata_np[0]
                    pre_processed_frame_right_np = outdata_np[1]
                else:
                    pre_processed_frame_left_np = pre_proc_func(new_frame_left_np)

        # Debugging
        # compare_array(pre_processed_frame_left, pre_processed_frame_left_np, offset)
        # if stereo:
        #     compare_array(pre_processed_frame_right, pre_processed_frame_right_np, offset)

        # Checking if 50% overlap to set the FFT input buffer
        if overlap == 50:
            if mode_list:
                if stereo:
                    input_frames_left[:block_size] = input_frames_left[block_size:]
                    input_frames_left[block_size:] = pre_processed_frame_left

                    input_frames_right[:block_size] = input_frames_right[block_size:]
                    input_frames_right[block_size:] = pre_processed_frame_right

                else:
                    input_frames_left[:block_size] = input_frames_left[block_size:]
                    input_frames_left[block_size:] = pre_processed_frame_left

            if mode_numpy:
                if stereo:
                    input_frames_left_np[:block_size] = input_frames_left_np[block_size:]
                    input_frames_left_np[block_size:] = pre_processed_frame_left_np

                    input_frames_right_np[:block_size] = input_frames_right_np[block_size:]
                    input_frames_right_np[block_size:] = pre_processed_frame_right_np

                else:
                    input_frames_left_np[:block_size] = input_frames_left_np[block_size:]
                    input_frames_left_np[block_size:] = pre_processed_frame_left_np

        # Checking if 75% overlap to set the FFT input buffer
        elif overlap == 75:
            if mode_list:
                if stereo:
                    input_frames_left[:block_size] = input_frames_left[block_size:2 * block_size]
                    input_frames_left[block_size:2 * block_size] = input_frames_left[2 * block_size:3 * block_size]
                    input_frames_left[2 * block_size:3 * block_size] = input_frames_left[3 * block_size:4 * block_size]
                    input_frames_left[3 * block_size:4 * block_size] = pre_processed_frame_left

                    input_frames_right[:block_size] = input_frames_right[block_size:2 * block_size]
                    input_frames_right[block_size:2 * block_size] = input_frames_right[2 * block_size:3 * block_size]
                    input_frames_right[2 * block_size:3 * block_size] = input_frames_right[
                                                                        3 * block_size:4 * block_size]
                    input_frames_right[3 * block_size:4 * block_size] = pre_processed_frame_right

                else:
                    input_frames_left[:block_size] = input_frames_left[block_size:2 * block_size]
                    input_frames_left[block_size:2 * block_size] = input_frames_left[2 * block_size:3 * block_size]
                    input_frames_left[2 * block_size:3 * block_size] = input_frames_left[3 * block_size:4 * block_size]
                    input_frames_left[3 * block_size:4 * block_size] = pre_processed_frame_left

            if mode_numpy:
                if stereo:
                    input_frames_left_np[:block_size] = input_frames_left_np[block_size:2 * block_size]
                    input_frames_left_np[block_size:2 * block_size] = input_frames_left_np[
                                                                      2 * block_size:3 * block_size]
                    input_frames_left_np[2 * block_size:3 * block_size] = input_frames_left_np[
                                                                          3 * block_size:4 * block_size]
                    input_frames_left_np[3 * block_size:4 * block_size] = pre_processed_frame_left_np

                    input_frames_right_np[:block_size] = input_frames_right_np[block_size:2 * block_size]
                    input_frames_right_np[block_size:2 * block_size] = input_frames_right_np[
                                                                       2 * block_size:3 * block_size]
                    input_frames_right_np[2 * block_size:3 * block_size] = input_frames_right_np[
                                                                           3 * block_size:4 * block_size]
                    input_frames_right_np[3 * block_size:4 * block_size] = pre_processed_frame_right_np
                else:
                    input_frames_left_np[:block_size] = input_frames_left_np[block_size:2 * block_size]
                    input_frames_left_np[block_size:2 * block_size] = input_frames_left_np[
                                                                      2 * block_size:3 * block_size]
                    input_frames_left_np[2 * block_size:3 * block_size] = input_frames_left_np[
                                                                          3 * block_size:4 * block_size]
                    input_frames_left_np[3 * block_size:4 * block_size] = pre_processed_frame_left_np

        # Debugging
        # compare_array(input_frames_left, input_frames_left_np, offset)
        # if stereo:
        #     compare_array(input_frames_right, input_frames_right_np, offset)

        # Applying the window function (Hanning)
        if mode_list:
            if stereo:
                for i in range(fft_frame_size):
                    fft_in_left[i] = input_frames_left[i] * hanning[i]
                    fft_in_right[i] = input_frames_right[i] * hanning[i]
            else:
                for i in range(fft_frame_size):
                    fft_in_left[i] = input_frames_left[i] * hanning[i]

        if mode_numpy:
            if stereo:
                # fft_in_left_np[:int(nfft / 2)] = input_frames_left_np * hanning_np
                # fft_in_right_np[:int(nfft / 2)] = input_frames_right_np * hanning_np
                fft_in_left_np[:fft_frame_size] = input_frames_left_np * hanning_np
                fft_in_right_np[:fft_frame_size] = input_frames_right_np * hanning_np
            else:
                # print(len(fft_in_left_np[:int(nfft / 2)]), len(input_frames_left_np), len(hanning_np))
                fft_in_left_np[:fft_frame_size] = input_frames_left_np * hanning_np

        # Debugging
        # compare_array(fft_in_left, fft_in_left_np, offset)
        # if stereo:
        #     compare_array(fft_in_right, fft_in_right_np, offset)

        fft_out_left = None
        fft_out_right = None
        fft_out_left_np = None
        fft_out_right_np = None

        # Doing the FFT operation
        if mode_list:
            fft_out_left = np.fft.fft(fft_in_left)
            if stereo:
                fft_out_right = np.fft.fft(fft_in_right)

        if mode_numpy:
            fft_out_left_np = np.fft.fft(fft_in_left_np)
            if stereo:
                fft_out_right_np = np.fft.fft(fft_in_right_np)

        # Debugging
        # compare_array(fft_out_left, fft_out_left_np, offset)
        # if stereo:
        #     compare_array(fft_out_right, fft_out_right_np, offset)

        fft_channels_left = None
        fft_channels_right = None
        fft_channels_left_np = None
        fft_channels_right_np = None

        # Revoming reflection
        if mode_list:
            fft_channels_left = fft_out_left[:(int(nfft / 2) + 1)]
            if stereo:
                fft_channels_right = fft_out_right[:(int(nfft / 2) + 1)]

        if mode_numpy:
            fft_channels_left_np = fft_out_left_np[:(int(nfft / 2) + 1)]
            if stereo:
                fft_channels_right_np = fft_out_right_np[:(int(nfft / 2) + 1)]

        # Debugging
        # compare_array(fft_channels_left, fft_channels_left_np, offset)
        # if stereo:
        #     compare_array(fft_channels_right, fft_channels_right_np, offset)

        freq_processed_frame_left = None
        freq_processed_frame_right = None
        freq_processed_frame_left_np = None
        freq_processed_frame_right_np = None

        # Checking if there is a frequency domain process function
        # No frequency domain process function just copies the input
        if not freq_proc_func:
            if mode_list:
                freq_processed_frame_left = fft_channels_left
                if stereo:
                    freq_processed_frame_right = fft_channels_right

            if mode_numpy:
                freq_processed_frame_left_np = fft_channels_left_np
                if stereo:
                    freq_processed_frame_right_np = fft_channels_right_np

        # Calling the frequency domain process function
        else:
            if mode_list:
                if stereo:
                    # Adding the 2 streams in one list
                    indata = [fft_channels_left, fft_channels_right]
                    outdata = freq_proc_func(indata)
                    freq_processed_frame_left = outdata[0]
                    freq_processed_frame_right = outdata[1]
                else:
                    freq_processed_frame_left = freq_proc_func(fft_channels_left)

            if mode_numpy:
                if stereo:
                    indata_np = np.empty((0, int(nfft / 2) + 1))
                    indata_np = np.append(indata_np, [fft_channels_left_np], axis=0)
                    indata_np = np.append(indata_np, [fft_channels_right_np], axis=0)
                    outdata_np = freq_proc_func(indata_np)

                    freq_processed_frame_left_np = outdata_np[0]
                    freq_processed_frame_right_np = outdata_np[1]
                else:
                    freq_processed_frame_left_np = freq_proc_func(fft_channels_left_np)

        # Debugging
        # compare_array(freq_processed_frame_left, freq_processed_frame_left_np, offset)
        # if stereo:
        #     compare_array(freq_processed_frame_right, freq_processed_frame_right_np, offset)

        # Mirror the spectrum
        if mode_list:
            if stereo:
                fft_out_left[:int(nfft / 2)] = freq_processed_frame_left[:int(nfft / 2)]
                fft_out_right[:int(nfft / 2)] = freq_processed_frame_right[:int(nfft / 2)]
                for i in range(int(nfft / 2)):
                    # fft_out_left[i]=freq_processed_frame_left[i]
                    fft_out_left[nfft - 1 - i] = np.conj(freq_processed_frame_left[i + 1])
                    # fft_out_right[i]=freq_processed_frame_right[i]
                    fft_out_right[nfft - 1 - i] = np.conj(freq_processed_frame_right[i + 1])

                # Nyquist
                fft_out_left[int(nfft / 2)] = freq_processed_frame_left[int(nfft / 2)]
                fft_out_right[int(nfft / 2)] = freq_processed_frame_right[int(nfft / 2)]
            else:
                for i in range(int(nfft / 2)):
                    fft_out_left[i] = freq_processed_frame_left[i]
                    fft_out_left[nfft - 1 - i] = np.conj(freq_processed_frame_left[i + 1])
                fft_out_left[int(nfft / 2)] = freq_processed_frame_left[int(nfft / 2)]

        if mode_numpy:
            if stereo:
                fft_out_left_np[:int(nfft / 2)] = freq_processed_frame_left_np[:int(nfft / 2)]
                temp_mirror = freq_processed_frame_left_np[1:int(nfft / 2)]
                temp_mirror = temp_mirror[::-1]
                fft_out_left_np[int(nfft / 2) + 1:] = np.conj(temp_mirror)

                fft_out_right_np[:int(nfft / 2)] = freq_processed_frame_right_np[:int(nfft / 2)]
                temp_mirror = freq_processed_frame_right_np[1:int(nfft / 2)]
                temp_mirror = temp_mirror[::-1]
                fft_out_right_np[int(nfft / 2) + 1:] = np.conj(temp_mirror)

                # Nyquist
                fft_out_left_np[int(nfft / 2)] = freq_processed_frame_left_np[int(nfft / 2)]
                fft_out_right_np[int(nfft / 2)] = freq_processed_frame_right_np[int(nfft / 2)]
            else:
                fft_out_left_np[:int(nfft / 2)] = freq_processed_frame_left_np[:int(nfft / 2)]
                temp_mirror = freq_processed_frame_left_np[1:int(nfft / 2)]
                temp_mirror = temp_mirror[::-1]
                fft_out_left_np[int(nfft / 2) + 1:] = np.conj(temp_mirror)

                # Nyquist
                fft_out_left_np[int(nfft / 2)] = freq_processed_frame_left_np[int(nfft / 2)]

        # Debugging
        # compare_array(fft_out_left, fft_out_left_np, offset)
        # if stereo:
        #     compare_array(fft_out_right, fft_out_right_np, offset)

        ifft_in_left = None
        ifft_in_right = None
        ifft_in_left_np = None
        ifft_in_right_np = None

        # Doing the IFFT operation
        if mode_list:
            ifft_in_left = np.fft.ifft(fft_out_left)
            if stereo:
                ifft_in_right = np.fft.ifft(fft_out_right)

        if mode_numpy:
            ifft_in_left_np = np.fft.ifft(fft_out_left_np)
            if stereo:
                ifft_in_right_np = np.fft.ifft(fft_out_right_np)

        # Debugging
        # compare_array(ifft_in_left, ifft_in_left_np, offset)
        # if stereo:
        #     compare_array(ifft_in_right, ifft_in_right_np, offset)

        # Separating the desired part of the IFFT output
        # For 50% overlap no second window needed
        if overlap == 50:
            if mode_list:
                if stereo:
                    ifft_out_left = [ifft_in_left[i].real for i in range(fft_frame_size)]
                    ifft_out_right = [ifft_in_right[i].real for i in range(fft_frame_size)]
                    out_frame_left = [ifft_out_left[i] + windowed_frame_left[i] for i in range(new_frame_size)]
                    windowed_frame_left = ifft_out_left[new_frame_size:]
                    out_frame_right = [ifft_out_right[i] + windowed_frame_right[i] for i in range(new_frame_size)]
                    windowed_frame_right = ifft_out_right[new_frame_size:]

                else:
                    ifft_out_left = [ifft_in_left[i].real for i in range(fft_frame_size)]
                    out_frame_left = [ifft_out_left[i] + windowed_frame_left[i] for i in range(new_frame_size)]
                    windowed_frame_left = ifft_out_left[new_frame_size:]

            if mode_numpy:
                if stereo:
                    ifft_out_left_np = ifft_in_left_np[:fft_frame_size].real
                    out_frame_left_np = ifft_out_left_np[:new_frame_size] + windowed_frame_left_np[:new_frame_size]
                    windowed_frame_left_np = ifft_out_left_np[new_frame_size:]

                    ifft_out_right_np = ifft_in_right_np[:fft_frame_size].real
                    out_frame_right_np = ifft_out_right_np[:new_frame_size] + windowed_frame_right_np[:new_frame_size]
                    windowed_frame_right_np = ifft_out_right_np[new_frame_size:]
                else:
                    ifft_out_left_np = ifft_in_left_np[:fft_frame_size].real
                    out_frame_left_np = ifft_out_left_np[:new_frame_size] + windowed_frame_left_np[:new_frame_size]
                    windowed_frame_left_np = ifft_out_left_np[new_frame_size:]

        # For 75% applying second window with gain adjustment to normalize output (-1< out <1)
        elif overlap == 75:
            if mode_list:
                if stereo:
                    ifft_out_left = [float(ifft_in_left[i].real) * hanning[i] * (2 / 3) for i in range(fft_frame_size)]
                    ifft_out_right = [float(ifft_in_right[i].real) * hanning[i] * (2 / 3) for i in
                                      range(fft_frame_size)]

                    out_frame_left = [ifft_out_left[i] + windowed_frame_left[i] for i in range(new_frame_size)]

                    windowed_frame_left[:new_frame_size] = [
                        windowed_frame_left[i + new_frame_size] + ifft_out_left[i + new_frame_size] for i in
                        range(new_frame_size)]

                    windowed_frame_left[new_frame_size:2 * new_frame_size] = [
                        windowed_frame_left[i + 2 * new_frame_size] + ifft_out_left[i + 2 * new_frame_size] for i in
                        range(new_frame_size)]

                    windowed_frame_left[2 * new_frame_size:] = ifft_out_left[3 * new_frame_size:]

                    out_frame_right = [ifft_out_right[i] + windowed_frame_right[i] for i in range(new_frame_size)]

                    windowed_frame_right[:new_frame_size] = [
                        windowed_frame_right[i + new_frame_size] + ifft_out_right[i + new_frame_size] for i in
                        range(new_frame_size)]

                    windowed_frame_right[new_frame_size:2 * new_frame_size] = [
                        windowed_frame_right[i + 2 * new_frame_size] + ifft_out_right[i + 2 * new_frame_size] for i in
                        range(new_frame_size)]

                    windowed_frame_right[2 * new_frame_size:] = ifft_out_right[3 * new_frame_size:]

                else:
                    ifft_out_left = [float(ifft_in_left[i].real) * hanning[i] * (2 / 3) for i in range(fft_frame_size)]

                    out_frame_left = [ifft_out_left[i] + windowed_frame_left[i] for i in range(new_frame_size)]
                    windowed_frame_left[:new_frame_size] = [
                        windowed_frame_left[i + new_frame_size] + ifft_out_left[i + new_frame_size] for i in
                        range(new_frame_size)]
                    windowed_frame_left[new_frame_size:2 * new_frame_size] = [
                        windowed_frame_left[i + 2 * new_frame_size] + ifft_out_left[i + 2 * new_frame_size] for i in
                        range(new_frame_size)]
                    windowed_frame_left[2 * new_frame_size:] = ifft_out_left[3 * new_frame_size:]

            if mode_numpy:
                if stereo:
                    ifft_out_left_np = np.float64(ifft_in_left_np[:fft_frame_size].real) * hanning_np * 2 / 3

                    out_frame_left_np = ifft_out_left_np[:new_frame_size] + windowed_frame_left_np[:new_frame_size]

                    windowed_frame_left_np[:new_frame_size] \
                        = windowed_frame_left_np[new_frame_size:2 * new_frame_size] \
                          + ifft_out_left_np[new_frame_size:2 * new_frame_size]

                    windowed_frame_left_np[new_frame_size:2 * new_frame_size] \
                        = windowed_frame_left_np[2 * new_frame_size:3 * new_frame_size] \
                          + ifft_out_left_np[2 * new_frame_size:3 * new_frame_size]

                    windowed_frame_left_np[2 * new_frame_size:] = ifft_out_left_np[3 * new_frame_size:]

                    ifft_out_right_np = np.float64(ifft_in_right_np[:fft_frame_size].real) * hanning_np * 2 / 3

                    out_frame_right_np = ifft_out_right_np[:new_frame_size] + windowed_frame_right_np[:new_frame_size]
                    windowed_frame_right_np[:new_frame_size] \
                        = windowed_frame_right_np[new_frame_size:2 * new_frame_size] \
                          + ifft_out_right_np[new_frame_size:2 * new_frame_size]

                    windowed_frame_right_np[new_frame_size:2 * new_frame_size] \
                        = windowed_frame_right_np[2 * new_frame_size:3 * new_frame_size] \
                          + ifft_out_right_np[2 * new_frame_size:3 * new_frame_size]

                    windowed_frame_right_np[2 * new_frame_size:] = ifft_out_right_np[3 * new_frame_size:]

                else:
                    ifft_out_left_np = np.float64(ifft_in_left_np[:fft_frame_size].real) * hanning_np * 2 / 3

                    out_frame_left_np = ifft_out_left_np[:new_frame_size] \
                                        + windowed_frame_left_np[:new_frame_size]

                    windowed_frame_left_np[:new_frame_size] \
                        = windowed_frame_left_np[new_frame_size:2 * new_frame_size] \
                          + ifft_out_left_np[new_frame_size:2 * new_frame_size]

                    windowed_frame_left_np[new_frame_size:2 * new_frame_size] \
                        = windowed_frame_left_np[2 * new_frame_size:3 * new_frame_size] \
                          + ifft_out_left_np[2 * new_frame_size:3 * new_frame_size]

                    windowed_frame_left_np[2 * new_frame_size:] = ifft_out_left_np[3 * new_frame_size:]

        # Debugging
        # compare_array(windowed_frame_left, windowed_frame_left_np, offset)
        # if stereo:
        #     compare_array(windowed_frame_right, windowed_frame_right_np, offset)

        # Checking if there is a post process function
        # No post process function just copies the input
        post_processed_frame_left = None
        post_processed_frame_right = None
        post_processed_frame_left_np = None
        post_processed_frame_right_np = None

        if not post_proc_func:
            if mode_list:
                post_processed_frame_left = out_frame_left
                if stereo:
                    post_processed_frame_right = out_frame_right

            if mode_numpy:
                post_processed_frame_left_np = out_frame_left_np
                if stereo:
                    post_processed_frame_right_np = out_frame_right_np

        # Calling the post process function
        else:
            # Adding the 2 streams in one list
            if mode_list:
                if stereo:
                    indata = [out_frame_left, out_frame_right]
                    outdata = post_proc_func(indata)
                    post_processed_frame_left = outdata[0]
                    post_processed_frame_right = outdata[1]

                else:
                    post_processed_frame_left = post_proc_func(out_frame_left)

            if mode_numpy:
                if stereo:
                    indata_np = np.empty((0, new_frame_size))
                    indata_np = np.append(indata_np, [out_frame_left_np], axis=0)
                    indata_np = np.append(indata_np, [out_frame_right_np], axis=0)
                    outdata = post_proc_func(indata_np)
                    post_processed_frame_left = outdata[0]
                    post_processed_frame_right = outdata[1]

                else:
                    post_processed_frame_left_np = post_proc_func(out_frame_left_np)

        # Debugging
        # compare_array(post_processed_frame_left, post_processed_frame_left_np, offset)
        # if stereo:
        #     compare_array(post_processed_frame_right, post_processed_frame_right_np, offset)

        # Adding the current frame to the output buffer
        if mode_list:
            for i in range(new_frame_size):
                output_left.append(post_processed_frame_left[i])
                if stereo:
                    output_right.append(post_processed_frame_right[i])

        if mode_numpy:
            # Adding the current frame to the output buffer
            # output_left_np = np.append(output_left_np, post_processed_frame_left_np[:new_frame_size])
            # if stereo:
            #     output_right_np = np.append(output_right_np, post_processed_frame_right_np[:new_frame_size])
            output_left_np[output_index_start:output_index_end] = post_processed_frame_left_np[:new_frame_size]
            if stereo:
                output_right_np[output_index_start:output_index_end] = post_processed_frame_right_np[:new_frame_size]

        # Debugging
        result_left = compare_array(np.array(output_left), np.array(output_left_np[:output_index_end]), offset)
        if stereo:
            result_right = compare_array(np.array(output_right), np.array(output_right_np[:output_index_end]), offset)

            if result_left == 1 and result_right == 1:
                result_list.append(True)
            else:
                # print(result_left, result_right)
                result_list.append(False)

        output_index_start += new_frame_size
        output_index_end += new_frame_size

        # Updating the progress bar
        if progress_bar:
            _print_progress(frame, len(frames_data_left))

        period_list.append(tm.perf_counter() - tic)

    print("\nProcess COMPLETE!")

    print(
        f"Total time: {sum(period_list)}, Max Period: {max(period_list)},"
        f" Average Period: {sum(period_list) / len(period_list)}")

    if mode_list and mode_numpy:
        if False in result_list:
            print("Convert FAIL!!")
            return None

        else:
            print("Convert SUCCESS!!")

    data = None
    # Return the output buffers Left and Right if out_file_name=""
    if out_file_name == "" or out_file_name is None:
        if stereo:
            outdata = [output_left, output_right]
            return outdata
        else:
            return output_left

    # Else save the output wav file
    else:
        left = np.asarray(output_left_np, dtype=np.float32)
        data = left
        if stereo:
            right = np.asarray(output_right_np, dtype=np.float32)
            data = np.vstack((left, right)).T
        wav.write(out_file_name, wav_freq, data)
