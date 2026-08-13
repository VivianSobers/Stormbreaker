"""Tests for saving a fitted model and reusing it against fresh data.

The hazard here is column alignment. A stored model is a vector of
coefficients whose meaning is positional, and the set of running applications
changes between runs. Scoring one application's activity against another's
coefficient would produce a confident, entirely wrong answer.
"""

import numpy as np
import pytest

from stormbreaker.model import (
    Dataset,
    align_to_fit,
    fit,
    load_fit,
    predict,
    save_fit,
    unknown_labels,
)
from stormbreaker.store import Store


def _dataset(columns, n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, len(columns))) * 1.5
    y = 3.0 + X.sum(axis=1) * 4.0 + rng.normal(0, 0.05, n)
    return Dataset(
        X=X,
        y=y,
        columns=list(columns),
        ts=np.arange(n, dtype=float),
        freq_edges=[],
        target="soc_w",
        win_ids=list(range(n)),
        globals_={"dt": np.full(n, 5.0), "discharging": np.zeros(n)},
    )


def test_fit_survives_a_round_trip(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    ds = _dataset([("a", "cpu"), ("b", "cpu")])
    original = fit(ds)

    save_fit(store, original)
    loaded, age = load_fit(store)

    np.testing.assert_allclose(loaded.coef, original.coef)
    assert loaded.baseline == pytest.approx(original.baseline)
    assert loaded.columns == original.columns
    assert loaded.target == original.target
    assert loaded.r2 == pytest.approx(original.r2)
    assert 0 <= age < 60
    store.close()


def test_no_saved_fit_returns_none(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    assert load_fit(store) is None
    store.close()


def test_alignment_survives_column_reordering():
    """The same applications in a different order must produce identical
    predictions, not a silently permuted attribution."""
    cols = [("a", "cpu"), ("b", "cpu")]
    ds = _dataset(cols)
    f = fit(ds)

    shuffled = Dataset(
        X=ds.X[:, ::-1].copy(),
        y=ds.y,
        columns=list(reversed(cols)),
        ts=ds.ts,
        freq_edges=[],
        target=ds.target,
        win_ids=ds.win_ids,
        globals_=ds.globals_,
    )
    aligned = align_to_fit(shuffled, f)

    assert aligned.columns == f.columns
    np.testing.assert_allclose(predict(aligned, f), predict(ds, f), rtol=1e-9)


def test_application_the_model_never_saw_is_dropped():
    """A newly launched application has no coefficient. It must contribute
    zero rather than borrow the coefficient sitting at its column index."""
    f = fit(_dataset([("a", "cpu"), ("b", "cpu")]))

    fresh = _dataset([("a", "cpu"), ("b", "cpu"), ("newcomer", "cpu")], seed=4)
    aligned = align_to_fit(fresh, f)

    assert aligned.columns == f.columns
    assert aligned.X.shape[1] == len(f.columns)
    assert unknown_labels(fresh, f) == {"newcomer"}


def test_application_missing_from_fresh_data_scores_zero():
    """An application in the model that is not running now contributes zero,
    and must not shift any other application's attribution."""
    f = fit(_dataset([("a", "cpu"), ("b", "cpu")]))

    fresh = _dataset([("a", "cpu")], seed=9)
    aligned = align_to_fit(fresh, f)

    b_col = f.columns.index(("b", "cpu"))
    assert np.all(aligned.X[:, b_col] == 0.0)
    a_col = f.columns.index(("a", "cpu"))
    np.testing.assert_allclose(aligned.X[:, a_col], fresh.X[:, 0])
