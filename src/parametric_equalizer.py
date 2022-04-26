import numpy as np
from scipy.signal import freqz


class ParametricEqualizer(object):
    def __init__(self, sampling_freq) -> None:
        self.sampling_freq = sampling_freq
        self._coeff = []

    @property
    def coeff(self):
        return tuple(self._coeff)

    @coeff.setter
    def coeff(self, value):
        if len(value) == 2:
            self._coeff.append((value[0], value[1], None))
        else:
            self._coeff.append(value)

    def freqz(self, full=False):
        """Apply parametric filter to data
                         -jw                  -jw              -jwM
                -jw   Bk(e  )    bk[0] + bk[1]e    + ... + bk[M]e
            H_k(e  ) = ------ = -----------------------------------
                         -jw                  -jw              -jwN
                    Ak(e  )    ak[0] + ak[1]e    + ... + ak[N]e
            
               -jwn               -jwn           
            H(e  ) =  \Pi H_k(e  )
        """
        fs = self.sampling_freq
        h = 1.0

        for b, a, _ in self._coeff:
            w, h_filter = freqz(b, a, worN=fs, include_nyquist=True, whole=full)
            h *= h_filter

        return w, h
