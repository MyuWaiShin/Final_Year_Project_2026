"""
tests/logger.py
---------------
Shared CSV logger for all pipeline test scripts.

Usage
-----
    from tests.logger import TestLogger

    log = TestLogger("nav")          # → results/nav_results.csv
    log.write(estimated_x=0.3, ...)
    log.close()
"""

import csv
import os
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


class TestLogger:
    """
    Appends one row per trial to a CSV file in tests/results/.
    First call creates the file and writes the header.

    Parameters
    ----------
    stage : str
        Short name for the test stage, e.g. "nav", "grasp", "verify",
        "transit", "recovery".  File will be  results/<stage>_results.csv.
    extra_fields : list[str], optional
        Additional column names beyond the standard ones.
    """

    # Columns present in every log file
    BASE_FIELDS = ["timestamp", "trial_id", "stage"]

    def __init__(self, stage: str, extra_fields: list = None):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.stage   = stage
        self.fields  = self.BASE_FIELDS + (extra_fields or [])
        self.path    = RESULTS_DIR / f"{stage}_results.csv"
        self._trial  = 0

        # Write header only if file is new / empty
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._fh  = open(self.path, "a", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._fh, fieldnames=self.fields,
                                   extrasaction="ignore")
        if write_header:
            self._csv.writeheader()

        print(f"[Logger] Logging {stage} results → {self.path}")

    def write(self, **kwargs):
        """
        Write one row.  Unrecognised keys are silently ignored (extrasaction='ignore').
        timestamp and trial_id are filled automatically if not supplied.
        """
        self._trial += 1
        row = {
            "timestamp": kwargs.pop("timestamp", datetime.now().isoformat(timespec="seconds")),
            "trial_id":  kwargs.pop("trial_id",  self._trial),
            "stage":     self.stage,
        }
        row.update(kwargs)
        self._csv.writerow(row)
        self._fh.flush()

    def close(self):
        self._fh.close()
        print(f"[Logger] Closed — {self._trial} rows written to {self.path}")

    # Context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
