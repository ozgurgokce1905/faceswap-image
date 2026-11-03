"""decord yerine geçen küçük paket (Python 3.12 için decord wheel yok).

Wan2.2 sadece VideoReader(path), len(), get_avg_fps(), get_frame_timestamp(-1),
get_batch(idxs).asnumpy() ve [i] kullanıyor; hepsi OpenCV ile sağlanıyor. RGB döner.
"""
import cv2
import numpy as np


class _Batch:
    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        return self._arr


class VideoReader:
    def __init__(self, path, ctx=None, width=-1, height=-1, num_threads=0):
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError('Video açılamadı: ' + path)
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cache = {}
        cap.release()

    def __len__(self):
        return self._n

    def get_avg_fps(self):
        return float(self._fps)

    def get_frame_timestamp(self, idx):
        if idx < 0:
            idx = self._n + idx
        t = idx / self._fps
        return np.array([t, (idx + 1) / self._fps], dtype=np.float32)

    def _read_all(self, wanted):
        wanted = sorted(set(int(i) for i in wanted))
        out = {}
        cap = cv2.VideoCapture(self.path)
        i = 0
        want_set = set(wanted)
        last = wanted[-1] if wanted else -1
        prev = None
        while i <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if i in want_set:
                out[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prev = out[i]
            i += 1
        cap.release()
        # eksik kareler (okunamayan son kareler) için son okunanı tekrarla
        for w in wanted:
            if w not in out:
                out[w] = prev if prev is not None else np.zeros((2, 2, 3), np.uint8)
        return out

    def get_batch(self, idxs):
        idxs = [int(i) for i in idxs]
        frames = self._read_all(idxs)
        return _Batch(np.stack([frames[i] for i in idxs], 0))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.get_batch(range(*idx.indices(self._n)))
        return _Batch(self._read_all([idx])[int(idx)])


def cpu(_=0):
    return None


def gpu(_=0):
    return None


class bridge:
    @staticmethod
    def set_bridge(name):
        return None
