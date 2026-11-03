
import numpy as np

SR = 16000
WINDOW_SECONDS = 4
HOP_SECONDS = 2

WIN = SR * WINDOW_SECONDS   # 64,000 samples
HOP = SR * HOP_SECONDS      # 32,000 samples


class RingBuffer:
    """Keep a rolling 4-second audio window in memory."""

    def __init__(self):
        self.buf = np.zeros(0, dtype=np.float32)
        self.new_samples = 0

    def push(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)

        if chunk.size == 0:
            return None

        # Append incoming samples and retain only the newest 4 seconds.
        self.buf = np.concatenate((self.buf, chunk))[-WIN:]
        self.new_samples += chunk.size

        # Wait for a full window and at least 2 seconds of new audio.
        if len(self.buf) == WIN and self.new_samples >= HOP:
            self.new_samples %= HOP
            return self.buf.copy()

        return None

    def clear(self):
        """Release buffered audio when the session ends."""
        self.buf = np.zeros(0, dtype=np.float32)
        self.new_samples = 0
