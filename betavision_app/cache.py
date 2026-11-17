"""
cache.py - one memory-bounded, thread-safe cache for everything a Job keeps per frame.

  * bounded by BYTES (numpy arrays are counted by nbytes), least-recently-used goes first,
    so scrubbing through any number of frames can't grow memory past the budget;
  * every access takes a lock, nothing is ever cleared wholesale, so one thread can't pull
    an entry out from under another;
  * single flight: if two threads ask for the same key at once, one computes it and the other
    waits for that result instead of computing it again.
"""

import sys
import threading
from collections import OrderedDict

import numpy as np

_SAMPLE = 64            # long lists are sized from their first items, not walked in full


def sizeof(obj, _seen=None, _depth=0):
    """Approximate bytes held by obj: arrays by nbytes (a buffer shared by several arrays is
    counted once), containers and objects by walking what they hold."""
    seen = set() if _seen is None else _seen
    if isinstance(obj, np.ndarray):
        base = obj
        while isinstance(base.base, np.ndarray):
            base = base.base
        if id(base) in seen:
            return 0
        seen.add(id(base))
        return int(base.nbytes)
    if _depth > 6 or id(obj) in seen:
        return 0
    if isinstance(obj, (str, bytes, int, float, bool, type(None))):
        return sys.getsizeof(obj)
    if isinstance(obj, dict):
        seen.add(id(obj))
        return sys.getsizeof(obj) + sum(sizeof(k, seen, _depth + 1) + sizeof(v, seen, _depth + 1)
                                        for k, v in obj.items())
    if isinstance(obj, (list, tuple, set, frozenset)):
        seen.add(id(obj))
        items = list(obj) if len(obj) <= _SAMPLE else list(obj)[:_SAMPLE]
        each = sum(sizeof(x, seen, _depth + 1) for x in items)
        if len(obj) > len(items):
            each = each * len(obj) / len(items)
        return sys.getsizeof(obj) + int(each)
    if hasattr(obj, "__dict__"):                  # e.g. climb_holds.FrameColour (its masks appear lazily)
        seen.add(id(obj))
        return sum(sizeof(v, seen, _depth + 1) for v in vars(obj).values())
    return sys.getsizeof(obj)


class ByteLRU:
    def __init__(self, max_bytes, size_of=sizeof):
        self.max_bytes = int(max_bytes)
        self._size_of = size_of
        self._items = OrderedDict()               # key -> value, least recently used first
        self._lock = threading.Lock()
        self._inflight = {}                       # key -> Event set when its computation ends

    def __len__(self):
        with self._lock:
            return len(self._items)

    def __contains__(self, key):
        with self._lock:
            return key in self._items

    def nbytes(self):
        """Current size. Entries can grow after they're stored (lazily computed masks), so
        this measures them now rather than trusting a size taken when they went in."""
        with self._lock:
            return sum(self._size_of(v) for v in self._items.values())

    def clear(self):
        with self._lock:
            self._items.clear()

    def get(self, key, default=None):
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        return default

    def get_or_compute(self, key, fn):
        """The value for key, computing it with fn() at most once however many threads ask at
        the same time. A result of None is returned but not stored (e.g. an unreadable frame);
        an exception goes to the thread that ran fn, and waiting threads try again."""
        while True:
            with self._lock:
                if key in self._items:
                    self._items.move_to_end(key)
                    return self._items[key]
                ev = self._inflight.get(key)
                if ev is None:
                    ev = self._inflight[key] = threading.Event()
                    mine = True
                else:
                    mine = False
            if not mine:
                ev.wait()
                continue
            try:
                value = fn()
                if value is not None:
                    self._put(key, value)
                return value
            finally:
                with self._lock:
                    self._inflight.pop(key, None)
                ev.set()

    def _put(self, key, value):
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            self._evict()

    def _evict(self):                              # caller holds the lock
        sizes = {k: self._size_of(v) for k, v in self._items.items()}
        total = sum(sizes.values())
        for k in list(self._items):                # oldest first; the newest entry may go too if it alone is over budget
            if total <= self.max_bytes:
                break
            del self._items[k]
            total -= sizes[k]
