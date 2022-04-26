"""Anaylzing the filter application

    This is for plotting the frequency response of the filter.
    It provides the parallel and cascades structure. The details of testing is on
    main.py.
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
from scipy.signal import freqz  # , freqs, tf2sos, sosfreqz, tf2zpk
import matplotlib.pyplot as plt

from .graphic_equalizer import GraphicalEqualizer
from .parametric_equalizer import ParametricEqualizer
from .util import (
    plot_frequency_response,
    plot_pole_zero_analysis,
)

from .config import DEBUG

if DEBUG:
    from .debugging import maker_logger

    PRINTER = maker_logger()


class _Graph(object):
    def __init__(self, sampling_freq, coeff, frequency_resp) -> None:
        self.sampling_freq = sampling_freq
        self.coeff = coeff
        self.frequency_resp = frequency_resp


class FilterAnalyzePlot(object):
    """Analyze filters using plot
    """

    def __init__(self, sampling_freq=None) -> None:
        self.graph_list = []
        self.sampling_freq = sampling_freq

    @property
    def filters(self):
        return self._filters

    @filters.setter
    def filters(self, *args):
        filter = args[0]
        if isinstance(filter, ParametricEqualizer):
            self.graph_list.append(
                _Graph(
                    sampling_freq=filter.sampling_freq,
                    coeff=filter.coeff,
                    frequency_resp=filter.freqz(),
                )
            )

        elif isinstance(filter, GraphicalEqualizer):
            self.graph_list.append(
                _Graph(
                    sampling_freq=filter.sampling_freq,
                    coeff=filter.coeff,
                    frequency_resp=filter.freqz(show=False),
                )
            )
        elif isinstance(filter, np.ndarray) or isinstance(filter, tuple):
            if self.sampling_freq is None:
                raise ValueError("sampling_freq must be given")
            else:
                if len(filter) == 2:
                    b, a = filter
                    d = None
                else:
                    b, a, d = filter
                self.graph_list.append(
                    _Graph(
                        sampling_freq=self.sampling_freq,
                        coeff=[(b, a, d),],
                        frequency_resp=freqz(
                            b=b,
                            a=a,
                            worN=self.sampling_freq,
                            whole=False,
                            include_nyquist=True,
                        ),
                    )
                )
        else:
            raise ValueError(
                "filter must be GraphicalEqualizer or ParametricEqualizer or b, a cofficient and sampling frequency"
            )

    def plot(self, type=["freq"], save_path=None, name=None):
        """Plot the filter
            type
                all: frequency response, phase response
                freq: frequency response
                phase: phase response
        """
        if len(self.graph_list) == 0:
            pass
        else:
            if save_path is None:
                pass
            else:
                if not isinstance(save_path, str):
                    raise ValueError("")
                self._savepath = save_path

            for graph in self.graph_list:
                fs = graph.sampling_freq
                (w, h) = graph.frequency_resp
                coeff = graph.coeff

                fticks = 1000 * 2.0 ** np.arange(-6, 5, 1)
                fticks = np.append(fticks, (fs / 2))
                fticklabels = [
                    "15.",
                    "31.",
                    "62.",
                    "125",
                    "250",
                    "500",
                    "1k",
                    "2k",
                    "4k",
                    "8k",
                    "16k",
                    str(int((fs / 2) // 1000)) + "k",
                ]
                flim = min(fticks), max(fticks)

                freq = fs * w / (2 * np.pi)
                h[np.argwhere(h == 0)] = -inf

                # _type = 0b001 : frequency response
                # _type = 0b010 : phase response
                # _type = 0b100 : zero-pole

                _type = 0
                ncols = 0
                if "freq" in type:
                    _type |= 0b001
                    ifreq = ncols
                    ncols += 1

                if "phase" in type:
                    _type |= 0b010
                    iphase = ncols
                    ncols += 1

                if "pole" in type:
                    _type |= 0b100
                    ipole = ncols
                    ncols += 1

                _fig = plt.figure(figsize=(ncols * 6, 7))
                _fig.gridspec_kw = {"wspace": 0.25}

                ax = [0] * ncols
                if _type & 0b001 == 0b001:
                    ax[ifreq] = _fig.add_subplot(1, ncols, ifreq + 1)
                if _type & 0b010 == 0b010:
                    ax[iphase] = _fig.add_subplot(1, ncols, iphase + 1)
                if _type & 0b100 == 0b100:
                    ax[ipole] = _fig.add_subplot(1, ncols, ipole + 1)

                if _type & 0b001 == 0b001:
                    plot_frequency_response(
                        ax[ifreq],
                        freq,
                        20 * np.log10(np.abs(h)),
                        xaxis=fticks,
                        xlim=flim,
                        xlabels=fticklabels,
                        xtitle="Frequency in Hz",
                        ytitle="Level in dB",
                    )

                if _type & 0b010 == 0b010:
                    plot_frequency_response(
                        ax[iphase],
                        freq,
                        np.angle(h, deg=True),
                        xaxis=fticks,
                        xlim=flim,
                        xlabels=fticklabels,
                        ylim=(-180, 180),
                        xtitle="Frequency in Hz",
                        ytitle="Phase in degree",
                    )

                if _type & 0b100 == 0b100:
                    if len(coeff) == 1:
                        b, a, d = coeff[0]
                    else:
                        size_filter = len(coeff)
                        b = np.zeros(shape=(3, size_filter))
                        a = np.zeros(shape=(3, size_filter))
                        for id_filter in range(size_filter):
                            _b, _a, _ = coeff[id_filter]
                            b[:, id_filter] = _b
                            a[:, id_filter] = _a
                    plot_pole_zero_analysis(ax[ipole], b, a)

                if name is not None:
                    plt.title(name)

                # Set the entire figure
                plt.get_current_fig_manager().full_screen_toggle()

                if save_path is None or save_path == "":
                    plt.show()
                    pass
                else:
                    plt.savefig(save_path)
