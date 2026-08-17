"""Tests for the collector loop.

The property that matters most here is that nothing the model does can take
the collector down. Sampling is irreplaceable — a window not recorded is gone
forever — whereas a fit can be recomputed from the data at any later time. So
a refit that fails must degrade to keeping the previous model and carrying on.
"""

from stormbreaker.caps import Caps
from stormbreaker.collect import Collector
from stormbreaker.model import load_fit


def _collector(tmp_path, **kw):
    # A blank Caps means no energy source and no cgroup root, which is the
    # harshest environment the collector can be asked to run in.
    return Collector(str(tmp_path / "c.db"), caps=Caps(), **kw)


def test_refit_on_empty_store_does_not_raise(tmp_path):
    c = _collector(tmp_path)
    msg = c.refit()
    assert msg is not None
    assert "skipped" in msg
    c.store.close()


def test_refit_failure_leaves_no_model_behind(tmp_path):
    c = _collector(tmp_path)
    c.refit()
    assert load_fit(c.store) is None
    c.store.close()


def test_collector_records_capabilities(tmp_path):
    c = _collector(tmp_path)
    assert c.store.get_meta("energy_target") == "none"
    assert c.store.get_meta("caps_json")
    c.store.close()


def test_refit_can_be_disabled(tmp_path):
    c = _collector(tmp_path, refit_every_s=0)
    assert c.refit_every_s == 0
    c.store.close()


def test_subsample_interval_never_exceeds_half_the_window(tmp_path):
    """A sub-sample interval longer than the window would collapse to endpoint
    sampling, which is what the averaging exists to avoid."""
    c = _collector(tmp_path, window_s=1.0, subsample_s=10.0)
    assert c.subsample_s <= 0.5
    c.store.close()


def test_collector_runs_outside_the_main_thread(tmp_path):
    """The collector is used as a library by the self-test, from a worker
    thread. Signal handlers can only be installed from the main thread, and
    registering them unconditionally killed that thread silently — no windows
    collected, no error surfaced.
    """
    import threading

    c = _collector(tmp_path, window_s=0.2, subsample_s=0.05)
    error: list[BaseException] = []

    def target():
        try:
            c.run(duration_s=0.6)
        except BaseException as e:  # noqa: BLE001 - the point is to catch all
            error.append(e)

    t = threading.Thread(target=target)
    t.start()
    t.join(timeout=20)

    assert not error, f"collector raised in a worker thread: {error}"
    assert not t.is_alive()
    c.store.close()


def test_memory_stat_is_off_by_default(tmp_path):
    """Page-fault collection is opt-in. Measured, it added ~54% to the scan
    cost while making held-out prediction slightly worse, so it does not get to
    cost every user something for nothing."""
    c = _collector(tmp_path)
    assert c.sampler.memory_stat is False
    c.store.close()


def test_memory_stat_can_be_enabled(tmp_path):
    from stormbreaker.caps import Caps
    from stormbreaker.collect import Collector

    caps = Caps()
    caps.has_memory_stat = True
    c = Collector(str(tmp_path / "m.db"), caps=caps, memory_stat=True)
    assert c.sampler.memory_stat is True
    c.store.close()


def test_memory_stat_stays_off_when_unsupported(tmp_path):
    """Asking for it on a kernel without memory.stat must not half-enable it."""
    from stormbreaker.caps import Caps
    from stormbreaker.collect import Collector

    caps = Caps()
    caps.has_memory_stat = False
    c = Collector(str(tmp_path / "m2.db"), caps=caps, memory_stat=True)
    assert c.sampler.memory_stat is False
    c.store.close()
