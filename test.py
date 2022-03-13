import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
t = np.arange(0, 6, 1/16000)
signal = np.sin(2*np.pi*t/2) + np.sin(2*np.pi*t/3)
signal_a = hilbert(signal)
plt.plot(t, signal, ".")
plt.plot(t, signal_a.imag, ".")
plt.plot(t, np.angle(signal_a), ".")
plt.show()

"""
pitch detection algorithm
- librosa.pyin
- pysptk
pysptk 나 pyin 과 같은 pitch detection algorithm을 사용하여 프레임별로 f0 를 추출하고, 원하시는 기준에 맞추어 일정 수준 이상의 값을 고르시는 방법
"""