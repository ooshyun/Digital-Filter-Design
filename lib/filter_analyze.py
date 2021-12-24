"""Anaylzing the filter application

    This is for plotting the frequency response of the filter.
    It provides the parallel and cascades structure. The details of testing is on
    test.py.
    ---
    TODO LIST
    - FilterAnalyzerPlot.plot
        [ ] 1. seperate serial equalizer structure
        [ ] 2. how to treat a phase in serail equalizer ?
        [ ] 3. Plot zero pole
        [ ] 4. graphical equalizer implementation
"""
from math import inf
import numpy as np
from scipy.signal import freqz #, freqs, tf2sos, sosfreqz, tf2zpk
import matplotlib.pyplot as plt

# from matplotlib.ticker import MultipleLocator
# from scipy.signal.filter_design import sos2tf

if __name__=='__main__':
    from util import plot_freqresp
else:
    from lib.util import plot_freqresp

class FilterAnalyzePlot(object):
    """Analyze filters using plot
    """
    def __init__(self, sampleing_freq) -> None:
        self._filters = []
        self.sampleing_freq = sampleing_freq
        self.graphical_equalizer = False
        self.series_equalizer = False

    @property
    def filters(self):
        return self._filters

    @filters.setter
    def filters(self, value):
        if isinstance(value, list):
            for val in value:
                self._filters.append(val)
        else:
            self._filters.append(np.array(value))

    def plot(self, type='all', save_path=None, name=None):
        """Plot the filter
            type
                all: frequency response, phase response
                freq: frequency response
                phase: phase response
        """
        if save_path is None:
            pass
        else:
            if not isinstance(save_path, str):
                raise ValueError('')
            self._savepath = save_path

        if len(self._filters) == 0:
            raise ValueError('There was NO dataset')

        w_total = []
        h_total = []

        fs = self.sampleing_freq
        fticks = 1000 * 2.**np.arange(-6, 5, 1)
        fticks = np.append(fticks, (fs/2))
        fticklabels = ['15.','31.', '62.','125', '250', '500', '1k', '2k', '4k','8k', '16k', str(int((fs/2)//1000))+'k']
        flim = min(fticks), max(fticks)

        # TODO: Seperate to another function 
        if self.series_equalizer:
            h=1.
            w=None
            for data in self._filters:
                b, a = data
                w, h_data = freqz(b, a, worN=2048*8) # TODO: How to treat phase ?
                h *= h_data
            w_total.append(w)
            h_total.append(h)
        else:
            for data in self._filters:
                b, a = data
                w, h = freqz(b, a, worN=2048*8)
                w_total.append(w)
                h_total.append(h)
        
        # Plot
        if type == 'all':
            ncols = 2
        else:
            ncols = 1

        _, ax = plt.subplots(figsize=(18, 7),
                    ncols=ncols,
                    gridspec_kw={'wspace': 0.25})

        for w, h in zip(w_total, h_total):
            freq=w * fs * 1.0 / (2 * np.pi)
            h[np.argwhere(h==0)] = -inf
            
            if type == 'all':
                plot_freqresp(ax[0], freq, 20*np.log(h), 
                        xaxis=fticks, xlim=flim,  xlabels=fticklabels,
                        xtitle='Frequency in Hz', ytitle='Level in dB')
                plot_freqresp(ax[1], freq, np.angle(h, deg=True), 
                         xaxis=fticks, xlim=flim,  xlabels=fticklabels,
                         ylim=(-180, 180), xtitle='Frequency in Hz', ytitle='Phase in degree')
            elif type=='gain':
                plot_freqresp(ax, freq, 20*np.log(h), 
                        xaxis=fticks, xlim=flim,  xlabels=fticklabels,
                        xtitle='Frequency in Hz', ytitle='Level in dB')
            elif type=='phase':
                plot_freqresp(ax, freq, np.angle(h, deg=True), 
                        xaxis=fticks, xlim=flim,  xlabels=fticklabels,
                        ylim=(-180, 180), xtitle='Frequency in Hz', ytitle='Phase in degree')
            else:
                raise ValueError('type must be gain or phase')

            # TODO: Plot zero pole
            # zpk = tf2zpk(b, a)
            # kw_z = dict(c='C0', marker='o', ms=9, ls='none', mew=1, mfc='none', alpha=1)
            # kw_p = dict(c='k', marker='x', ms=9, ls='none', mew=1)
            # z, p, _ = zpk
            # ax[2].plot(np.real(p), np.imag(p), **kw_p)
            # ax[2].plot(np.real(z), np.imag(z), **kw_z)

            # kw_artist = dict(edgecolor='gray', linestyle='-', linewidth=1)
            # circle = plt.Circle(xy=(0, 0), radius=1, facecolor='none', **kw_artist)
            # ax[2].add_artist(circle)
            # ax[2].axis([-1.2, 1.2, -1.2, 1.2])
            # ax[2].grid(True)
            # ax[2].set_aspect('equal')
            # ax[2].set_ylabel(r'$\Im (z)$')
            # ax[2].set_xlabel(r'$\Re (z)$')

            # ax[2].xaxis.set_major_locator(MultipleLocator(0.1))
            # ax[2].yaxis.set_major_locator(MultipleLocator(0.1))


        if name is not None:
            plt.title(name)

        plt.get_current_fig_manager().full_screen_toggle()
        if save_path is None or save_path == '':
            plt.show()
        else:
            plt.savefig(save_path)

if __name__=='__main__':
    from filt import CustomFilter
    fs = 44100
    fc = 1000

    ploter = FilterAnalyzePlot(sampleing_freq=fs)
    
    """ Custom filter analysis"""
    filter_custom = CustomFilter(sampling_freq=fs,
                                cutoff_freq=fc)
    b_coeff, a_coeff = filter_custom.highpass()
    print(f"Cofficient: {b_coeff, a_coeff}")

    ploter.filters = b_coeff, a_coeff
    ploter.plot()