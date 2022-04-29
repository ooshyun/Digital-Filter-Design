"""Filter Design Libraray
    - biquad_cookbook: Single Filter
    - ParametricEqualizer: Cascade-Structure Fitler using Parametric equalizer
    - GraphicalEqualizer: Parallel-Structure Fitler using Graphical equalizer
    - WaveProcessor: Signal Processing with wav file and [TODO] mic streaming
    - FilterAnalyzePlot: Signal Plotting with designed filter
"""
from .graphic_equalizer import GraphicalEqualizer
from .parametric_equalizer import ParametricEqualizer
from .filter_analyze import FilterAnalyzePlot
from .filter_application import WaveProcessor
from .util import (
    cvt_float2fixed,
    cvt_pcm2wav,
    cvt_char2num,
    fi,
)
from .biquad_cookbook import lowpass, highpass, bandpass, notch, allpass, peaking, shelf
from .debugging import maker_logger, check_time, print_func_time
from .config import DEBUG, EPS

__all__ = [
    "GraphicalEqualizer",
    "ParametricEqualizer",
    "FilterAnalyzePlot",
    "WaveProcessor",
    "lowpass",
    "highpass",
    "bandpass",
    "notch",
    "allpass",
    "peaking",
    "shelf",
    "cvt_float2fixed",
    "cvt_pcm2wav",
    "cvt_char2num",
    "fi",
    "maker_logger",
    "check_time",
    "print_func_time",
    "DEBUG",
    "EPS",
]
