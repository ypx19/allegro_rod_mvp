"""Short frame-stack observation history for feed-forward policies.

Newest frame is first: stack_t = [obs_t, obs_{t-1}, ..., obs_{t-H+1}].
On reset, missing history is filled by repeating the initial observation.
history_len=1 is a no-op identity (same as the current single-frame obs).
"""

from __future__ import annotations

import numpy as np


class ObservationHistoryBuffer:
    """Per-environment observation ring used by `RodRotationEnv`.

    Vectorized trainers keep one buffer per env (SubprocVecEnv process or
    DummyVecEnv instance), so histories cannot leak across workers.
    """

    def __init__(self, history_len: int, frame_dim: int) -> None:
        if int(history_len) < 1:
            raise ValueError("history_len must be >= 1")
        if int(frame_dim) < 1:
            raise ValueError("frame_dim must be >= 1")
        self.history_len = int(history_len)
        self.frame_dim = int(frame_dim)
        self._frames = np.zeros((self.history_len, self.frame_dim), dtype=np.float32)
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def stacked_dim(self) -> int:
        return self.history_len * self.frame_dim

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """Fill every slot with `frame` and return the stacked observation."""
        frame_arr = np.asarray(frame, dtype=np.float32).reshape(self.frame_dim)
        self._frames[:] = frame_arr
        self._initialized = True
        return self.stacked()

    def push(self, frame: np.ndarray) -> np.ndarray:
        """Shift older frames back, write the newest frame at index 0."""
        frame_arr = np.asarray(frame, dtype=np.float32).reshape(self.frame_dim)
        if not self._initialized:
            return self.reset(frame_arr)
        if self.history_len > 1:
            self._frames[1:] = self._frames[:-1]
        self._frames[0] = frame_arr
        return self.stacked()

    def stacked(self) -> np.ndarray:
        return self._frames.reshape(-1).copy()

    def frames(self) -> np.ndarray:
        """Return a copy of the (history_len, frame_dim) buffer, newest first."""
        return self._frames.copy()
