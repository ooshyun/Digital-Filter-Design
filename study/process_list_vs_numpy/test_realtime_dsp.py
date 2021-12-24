"""
* @file     test_realtimedsp.py
* @brief    Tested different frame type changed list to numpy
* @details
    1. Vertifying convert
        Comment in real_time_dsp 
            # Debugging

    2. Processing time in a frame
        1. Total time
            List: 73.4591 sec
            Numpy: 3.6539 sec
        2. Max Period
            List: 0.2016 sec
            Numpy: 0.0054 sec
        3. Average Period
            List: 0.0045 sec
            Numpy: 0.0002 sec
            
* @author    daniel-oh
* @date      2021/06/15
"""

import real_time_dsp
import os

path_stereo = None
path_mono = None

test_case = {"in_file_name": [(path_stereo, True)],
             "overlap": (75, 50),
             "block_size": (128, 256, 512, 1024, 2048),
             "zero_pad": (True, False)
             }


def test():
    for path, stereo_type in test_case["in_file_name"]:
        for overlap in test_case["overlap"]:
            for block_size in test_case["block_size"]:
                for zero_pad in test_case["zero_pad"]:
                    print(f"Case\n"
                          f"Path: {path}\n"
                          f"StereoType: {stereo_type}\n"
                          f"overlap size: {overlap}\n"
                          f"block size: {block_size}\n"
                          f"Zero pad: {zero_pad}")

                    if not real_time_dsp.wave_file_process(in_file_name=path,
                                                           get_file_details=False,
                                                           out_file_name="",
                                                           progress_bar=True,
                                                           stereo=stereo_type,
                                                           overlap=overlap,
                                                           block_size=block_size,
                                                           zero_pad=zero_pad,
                                                           pre_proc_func=None,
                                                           freq_proc_func=None,
                                                           post_proc_func=None
                                                           ):
                        return path, stereo_type, overlap, block_size, zero_pad
                    print("----------------------Done----------------------")

    return None


if __name__ == "__main__":
    if test():
        print(test())
