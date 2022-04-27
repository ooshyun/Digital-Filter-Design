import numpy as np
import queue


class AudioBuffer(object):
    def __init__(self, frame_size) -> None:
        self._inbuffer = queue.Queue()
        self._outbuffer = queue.Queue()
        self._empty_frame = np.zeros(frame_size)

    @property
    def inbuffer(self):
        return self._inbuffer

    @inbuffer.setter
    def inbuffer(self, frame):
        self._inbuffer.put(frame)

    @inbuffer.getter
    def inbuffer(self):
        try:
            data = self.inbuffer.get_nowait()
        except queue.Empty:
            data = self._empty_frame.copy()
        return data

    @property
    def outbuffer(self):
        return self._outbuffer

    @outbuffer.setter
    def outbuffer(self, frame):
        self._outbuffer.put(frame)

    @outbuffer.getter
    def outbuffer(self):
        try:
            data = self.outbuffer.get_nowait()
        except queue.Empty:
            data = self._empty_frame.copy()
        return data
