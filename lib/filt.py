"""Parametric Filter Application
    
    This implementation is for Parametric filter
    It has several parameters, named Q factor, frequency component and gain.
        Gain step is 0.1 and max +12db, min -12db.
        Frequency range to be possible to change is - to -.
        Q factor range to be possible to change is - to -.
    It also analyzes the frequency response and pole, zero plot
    automatically using - and - function.

    TODO: shelf Q, S, BW comments from
    https://github.com/WebAudio/Audio-EQ-Cookbook/blob/main/Audio-EQ-Cookbook.txt
"""
from scipy.signal import iirfilter #, freqs
from math import sqrt
# import numpy as np

if __name__=='__main__':
    import biquad_cookbook
else:
    import lib.biquad_cookbook as biquad_cookbook

class CustomFilter(object):
    """Custom Filter Class
    """
    def __init__(self,
                sampling_freq,
                cutoff_freq,
                output_type='ba',
                analog=False,
                gain=None,
                Qfactor=None,
                bandwidth=None,
                slide=None) -> None:
        self.sampling_freq = sampling_freq
        self._cutoff_freq = cutoff_freq
        self.wn = 2 * self.cutoff_freq / self.sampling_freq
        self.output = output_type
        self.analog = analog

        self.gain = gain
        if Qfactor is None:
            print('Warning: Qfactor or Bandwidth or Slide are not set,'
                'Q set to 1/sqrt(1)')
            self.qfactor = 1/sqrt(2)
        else:
            self.qfactor = Qfactor
        self.bandwidth = bandwidth
        self.slide = slide

        self._parameter = None
        self._savepath = None

    @property
    def paramter(self):
        return self._parameter

    @property
    def cutoff_freq(self):
        return self._cutoff_freq

    @cutoff_freq.setter
    def cutoff_freq(self, value):
        self._cutoff_freq = value
        self.wn = 2 * self._cutoff_freq / self.sampling_freq

    def lowpass(self, filter_type='default'):
        """Lowpass filter
        """
        if filter_type == 'default':
            self._parameter = biquad_cookbook.lowpass(Wn=self.wn,
                                                    Q=self.qfactor,
                                                    analog=self.analog,
                                                    output=self.output)
        elif filter_type == 'iir':
            self._parameter = iirfilter(N=2,
                                    Wn=self.wn,
                                    btype='low',
                                    analog=False,
                                    ftype='butter',
                                    output=self.output)
        return self._parameter
    
    def highpass(self, filter_type='default'):
        """Highpass filter
        """
        if filter_type == 'default':
            self._parameter =  biquad_cookbook.highpass(Wn=self.wn,
                                                    Q=self.qfactor,
                                                    analog=self.analog,
                                                    output=self.output)
        elif filter_type == 'iir':
            self._parameter = iirfilter(N=2,
                                    Wn=self.wn,
                                    btype='low',
                                    analog=False,
                                    ftype='butter',
                                    output=self.output)
        return self._parameter

    def bandpass(self):
        """Bandpass filter
        TODO: fix to [a, b] bandpass filter
        TODO: warnings.warn("Badly conditioned filter coefficients (numerator) 
            in default
        """
        self._parameter = biquad_cookbook.bandpass(Wn=self.wn,
                                                Q=self.qfactor,
                                                type='skirt',
                                                analog=self.analog,
                                                output=self.output)
        return self._parameter


    def notch(self):
        """Notch filter
        """
        self._parameter = biquad_cookbook.notch(Wn=self.wn,
                                            Q=self.qfactor,
                                            analog=self.analog,
                                            output=self.output)
        return self._parameter

    def allpass(self):
        """Allpass filter
        """
        self._parameter = biquad_cookbook.allpass(Wn=self.wn,
                                            Q=self.qfactor,
                                            analog=self.analog,
                                            output=self.output)
        return self._parameter

    def peaking(self):
        """Peak filter
        TODO: not finished
        """
        if self.gain is None:
            self.gain = 0
            print("Warning: gain is not set, set to 1")
        self._parameter = biquad_cookbook.peaking(Wn=self.wn,
                                            dBgain=self.gain,
                                            Q=self.qfactor,
                                            BW=None,
                                            type='half',
                                            analog=self.analog,
                                            output=self.output)
        return self._parameter

    def shelf(self):
        """Shelving filter
        """
        if self.gain is None:
            self.gain = 0
            print('Warning: gain is not set, set to 1')

        if self.qfactor is None and self.bandwidth is None \
                                and self.slide is None:
            self.qfactor = 1/sqrt(2)
            print('Warning: Qfactor or Bandwidth or Slide are not set,'
                'Q set to 1/sqrt(1)')

        self._parameter = biquad_cookbook.shelf(Wn=self.wn,
                                            dBgain=self.gain,
                                            Q=self.qfactor,
                                            S=self.slide,
                                            BW=self.bandwidth,
                                            btype='low',
                                            ftype='half',
                                            analog=self.analog,
                                            output=self.output)
        return self._parameter

if __name__ == '__main__':
    from filter_analyze import FilterAnalyzePlot
    fs = 44100
    fc = 1000

    filter_custom = CustomFilter(sampling_freq=fs,
                                cutoff_freq=fc)

    ploter = FilterAnalyzePlot(sampleing_freq=fs)
    ploter.filters = filter_custom.highpass()
    ploter.plot()

    