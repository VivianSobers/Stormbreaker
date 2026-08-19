"""SQLite storage for the counter time-series.

One row per window in ``window``, one row per (window, cgroup) in ``sample``.
At a 5 s window with ~40 active cgroups this is roughly 2-3 MB per week, which
is the budget the design calls for. Everything stays on the local disk; nothing
is uploaded anywhere.
"""

from __future__ import annotations

import os
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS label (
    id   INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS window (
    id           INTEGER PRIMARY KEY,
    ts           REAL NOT NULL,     -- wall clock at end of window
    dt           REAL NOT NULL,     -- window duration, seconds
    soc_w        REAL,              -- package/SoC watts (hwmon)
    rapl_pkg_w   REAL,              -- RAPL package watts, if readable
    rapl_core_w  REAL,              -- RAPL core watts, if readable
    gpu_busy     REAL,              -- 0..1
    freq_ghz     REAL,
    batt_w       REAL,              -- +discharge / -charge
    discharging  INTEGER,
    charge       REAL,              -- uAh or uWh, battery dependent
    volt_v       REAL,              -- pack voltage; needed to turn uAh into Wh
    temp_c       REAL,              -- package temperature; leakage rises with it
    profile      TEXT               -- power regime; a change invalidates coefficients
);
CREATE INDEX IF NOT EXISTS window_ts ON window(ts);

CREATE TABLE IF NOT EXISTS sample (
    win_id   INTEGER NOT NULL REFERENCES window(id) ON DELETE CASCADE,
    label_id INTEGER NOT NULL REFERENCES label(id),
    cpu      REAL,
    io_mb    REAL,
    ctxt_k   REAL,
    gpu      REAL,
    pgflt_k  REAL,
    nr_procs REAL,
    PRIMARY KEY (win_id, label_id)
) WITHOUT ROWID;
"""

# Columns added after the first release. SQLite's CREATE TABLE IF NOT EXISTS
# does nothing to a table that already exists, so a schema change silently
# leaves old databases unreadable by new code — and a recording costs hours of
# wall time to reproduce. Every added column is listed here and applied on open.
MIGRATIONS: dict[str, list[tuple[str, str]]] = {
    "window": [
        ("volt_v", "REAL"),
        ("temp_c", "REAL"),
        ("profile", "TEXT"),
    ],
    "sample": [
        ("pgflt_k", "REAL"),
    ],
}

DEFAULT_DB = os.path.join(
    os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
    "stormbreaker",
    "stormbreaker.db",
)


class Store:
    def __init__(self, path: str = DEFAULT_DB):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # check_same_thread=False because the collector is also driven from a
        # worker thread (see selftest), and Python's sqlite3 is built in
        # serialized mode, so the library locks internally. Access here is in
        # any case effectively serialised: exactly one thread writes.
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        self._migrate()
        self.db.commit()
        self._labels: dict[str, int] = {
            r["name"]: r["id"] for r in self.db.execute("SELECT id, name FROM label")
        }

    def _migrate(self) -> None:
        """Add any columns an older database is missing.

        Cheap (SQLite rewrites only the header), idempotent, and it means a
        recording made by any earlier version stays readable rather than
        having to be collected again.
        """
        for table, cols in MIGRATIONS.items():
            have = {
                r[1] for r in self.db.execute(f"PRAGMA table_info({table})")
            }
            if not have:
                continue
            for name, decl in cols:
                if name not in have:
                    self.db.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {decl}"
                    )

    # -- writing --------------------------------------------------------------

    def label_id(self, name: str) -> int:
        hit = self._labels.get(name)
        if hit is not None:
            return hit
        cur = self.db.execute("INSERT OR IGNORE INTO label(name) VALUES (?)", (name,))
        if cur.lastrowid and cur.rowcount:
            new_id = cur.lastrowid
        else:
            new_id = self.db.execute(
                "SELECT id FROM label WHERE name = ?", (name,)
            ).fetchone()["id"]
        self._labels[name] = new_id
        return new_id

    def add_window(
        self, g: dict[str, float], feats: dict[str, dict[str, float]]
    ) -> int:
        cur = self.db.execute(
            """INSERT INTO window
               (ts, dt, soc_w, rapl_pkg_w, rapl_core_w, gpu_busy,
                freq_ghz, batt_w, discharging, charge, volt_v, temp_c, profile)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(),
                g.get("dt"),
                g.get("soc_w"),
                _first(g, "rapl_package-0_w", "rapl_package_w"),
                _first(g, "rapl_core_w"),
                g.get("gpu_busy"),
                g.get("freq_ghz"),
                g.get("batt_w"),
                int(g.get("discharging", 0)),
                g.get("charge"),
                g.get("volt_v"),
                g.get("temp_c"),
                g.get("profile"),
            ),
        )
        wid = cur.lastrowid
        rows = [
            (
                wid,
                self.label_id(label),
                f["cpu"],
                f["io_mb"],
                f["ctxt_k"],
                f["gpu"],
                f.get("pgflt_k", 0.0),
                f["nr_procs"],
            )
            for label, f in feats.items()
        ]
        self.db.executemany(
            "INSERT OR REPLACE INTO sample VALUES (?,?,?,?,?,?,?,?)", rows
        )
        return wid

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (key, value)
        )

    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def commit(self) -> None:
        self.db.commit()

    def prune(self, keep_days: float) -> int:
        cutoff = time.time() - keep_days * 86400
        cur = self.db.execute("DELETE FROM window WHERE ts < ?", (cutoff,))
        self.db.commit()
        return cur.rowcount

    # -- reading --------------------------------------------------------------

    def window_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) c FROM window").fetchone()["c"]

    def windows(self, since: float | None = None, limit: int | None = None):
        sql = "SELECT * FROM window"
        args: list = []
        if since is not None:
            sql += " WHERE ts >= ?"
            args.append(since)
        sql += " ORDER BY ts"
        if limit:
            sql = (
                f"SELECT * FROM ({sql.replace('ORDER BY ts', 'ORDER BY ts DESC')} "
                f"LIMIT {int(limit)}) ORDER BY ts"
            )
        return self.db.execute(sql, args).fetchall()

    def samples_for(self, win_ids: list[int]):
        """Per-window feature rows for the given windows, joined to labels."""
        out: dict[int, dict[str, dict[str, float]]] = {}
        chunk = 900  # stay under SQLITE_MAX_VARIABLE_NUMBER
        for i in range(0, len(win_ids), chunk):
            part = win_ids[i : i + chunk]
            q = ",".join("?" * len(part))
            rows = self.db.execute(
                f"""SELECT s.win_id, l.name, s.cpu, s.io_mb, s.ctxt_k, s.gpu,
                           s.pgflt_k, s.nr_procs
                    FROM sample s JOIN label l ON l.id = s.label_id
                    WHERE s.win_id IN ({q})""",
                part,
            )
            for r in rows:
                out.setdefault(r["win_id"], {})[r["name"]] = {
                    "cpu": r["cpu"],
                    "io_mb": r["io_mb"],
                    "ctxt_k": r["ctxt_k"],
                    "gpu": r["gpu"],
                    "pgflt_k": r["pgflt_k"] or 0.0,
                    "nr_procs": r["nr_procs"],
                }
        return out

    def close(self) -> None:
        self.db.commit()
        self.db.close()


def _first(d: dict[str, float], *keys: str) -> float | None:
    for k in keys:
        if k in d:
            return d[k]
    return None
