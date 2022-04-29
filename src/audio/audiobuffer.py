"""Buffer for Sequencial dataset such as Audio and Speech

    This represents the buffer that is used to store the audio data, queue format
"""
import numpy as np
import queue


class AudioBuffer(object):
    def __init__(self, frame_size: int) -> None:
        """Provide and Save the frame data to the buffer
    
        Parameters
        ----------
            frame_size (int): The size of the frame when processing the data 
        """
        self._inbuffer = queue.Queue()
        self._outbuffer = queue.Queue()
        self._empty_frame = np.zeros(frame_size)

    @property
    def inbuffer(self):
        return self._inbuffer

    @inbuffer.setter
    def inbuffer(self, frame: np.ndarray):
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
