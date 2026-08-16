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
