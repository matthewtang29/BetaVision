import threading
import time

import numpy as np
import pytest

from betavision_app.cache import ByteLRU, sizeof


def arr(mb):
    return np.zeros(int(mb * 1024 * 1024), np.uint8)


def test_bounded_by_bytes_not_entries():
    c = ByteLRU(10 * 1024 * 1024)
    for i in range(200):                       # scrub through 200 "frames" of 3 MB each
        c.get_or_compute(i, lambda: arr(3))
        assert c.nbytes() <= 10 * 1024 * 1024
    assert 1 <= len(c) <= 3


def test_least_recently_used_goes_first():
    c = ByteLRU(7 * 1024 * 1024)
    c.get_or_compute("a", lambda: arr(3))
    c.get_or_compute("b", lambda: arr(3))
    c.get("a")                                  # a is now newer than b
    c.get_or_compute("c", lambda: arr(3))
    assert "a" in c and "c" in c and "b" not in c


def test_item_bigger_than_budget_is_returned_but_not_kept():
    c = ByteLRU(1024 * 1024)
    big = c.get_or_compute("big", lambda: arr(4))
    assert big.nbytes == 4 * 1024 * 1024 and len(c) == 0


def test_growth_after_insert_is_counted():
    class Lazy:                                 # like FrameColour: masks appear after it was stored
        pass
    c = ByteLRU(5 * 1024 * 1024)
    a = c.get_or_compute("a", lambda: Lazy())
    c.get_or_compute("b", lambda: arr(1))
    a.mask = arr(4.5)                           # grows lazily
    c.get_or_compute("c", lambda: arr(1))       # next insert re-measures and evicts
    assert c.nbytes() <= 5 * 1024 * 1024 and "a" not in c


def test_views_of_one_buffer_count_once():
    base = arr(2)
    assert sizeof({"x": base, "y": base[10:], "z": base.reshape(-1)}) < 2.1 * 1024 * 1024


def test_same_key_is_computed_once_across_threads():
    c = ByteLRU(10 * 1024 * 1024)
    calls, start = [], threading.Barrier(8)

    def slow():
        calls.append(1)
        time.sleep(0.2)
        return arr(1)

    results = []

    def worker():
        start.wait()
        results.append(c.get_or_compute("k", slow))
    ts = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(calls) == 1
    assert len(results) == 8 and all(r is results[0] for r in results)


def test_no_keyerror_when_many_threads_churn_a_tiny_cache():
    """The old dict + clear() raised KeyError when another thread cleared between the check and the read."""
    c = ByteLRU(2 * 1024 * 1024)
    errors = []

    def churn(seed):
        rng = np.random.default_rng(seed)
        try:
            for _ in range(300):
                k = int(rng.integers(0, 12))
                v = c.get_or_compute(k, lambda k=k: np.full(300_000, k, np.uint8))
                assert v[0] == k
        except Exception as e:                  # noqa: BLE001
            errors.append(e)
    ts = [threading.Thread(target=churn, args=(s,)) for s in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors[:1]
    assert c.nbytes() <= 2 * 1024 * 1024


def test_failed_computation_is_not_cached_and_can_be_retried():
    c = ByteLRU(1024 * 1024)

    def boom():
        raise RuntimeError("x")
    with pytest.raises(RuntimeError):
        c.get_or_compute("k", boom)
    assert c.get_or_compute("k", lambda: 5) == 5


def test_none_is_not_stored():
    c = ByteLRU(1024 * 1024)
    n = []
    assert c.get_or_compute("k", lambda: n.append(1)) is None
    assert c.get_or_compute("k", lambda: n.append(1)) is None
    assert len(n) == 2 and len(c) == 0
