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


def test_old_database_is_migrated_not_broken(tmp_path):
    """A schema change must never strand an existing recording.

    CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so
    adding a column silently left old databases unreadable by new code — and a
    recording costs hours of wall time to reproduce.
    """
    import sqlite3

    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE label (id INTEGER PRIMARY KEY, name TEXT UNIQUE);
        CREATE TABLE window (
            id INTEGER PRIMARY KEY, ts REAL NOT NULL, dt REAL NOT NULL,
            soc_w REAL, rapl_pkg_w REAL, rapl_core_w REAL, gpu_busy REAL,
            freq_ghz REAL, batt_w REAL, discharging INTEGER, charge REAL
        );
        CREATE TABLE sample (
            win_id INTEGER NOT NULL, label_id INTEGER NOT NULL,
            cpu REAL, io_mb REAL, ctxt_k REAL, gpu REAL, nr_procs REAL,
            PRIMARY KEY (win_id, label_id)
        ) WITHOUT ROWID;
        """
    )
    old.execute("INSERT INTO window(id, ts, dt, soc_w) VALUES (1, 100.0, 5.0, 7.5)")
    old.execute("INSERT INTO label(id, name) VALUES (1, 'firefox')")
    old.execute("INSERT INTO sample VALUES (1, 1, 0.5, 0.0, 0.0, 0.0, 3)")
    old.commit()
    old.close()

    store = Store(path)
    window_cols = {r[1] for r in store.db.execute("PRAGMA table_info(window)")}
    sample_cols = {r[1] for r in store.db.execute("PRAGMA table_info(sample)")}

    assert {"volt_v", "temp_c", "profile"} <= window_cols
    assert "pgflt_k" in sample_cols

    # and the pre-existing row survived untouched
    assert store.window_count() == 1
    row = store.db.execute("SELECT soc_w FROM window WHERE id=1").fetchone()
    assert row["soc_w"] == pytest.approx(7.5)
    samples = store.samples_for([1])
    assert samples[1]["firefox"]["cpu"] == pytest.approx(0.5)
    store.close()


def test_migration_is_idempotent(tmp_path):
    path = str(tmp_path / "twice.db")
    Store(path).close()
    s = Store(path)          # opening again must not fail on duplicate columns
    assert s.window_count() == 0
    s.close()
