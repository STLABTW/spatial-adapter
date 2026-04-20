"""
Shared utilities for experiment scripts.
"""

import csv
import functools
import time
from pathlib import Path
from typing import Optional


class TimerLog:
    """Accumulates wall-time records and writes them to CSV.

    Usage::

        log = TimerLog("stage2_timings.csv")

        @log.timer(seed=42, model="reg")
        def run_stage2(trainer):
            trainer.run()

        # ... after all seeds ...
        log.save()          # writes CSV
        log.summary()       # prints mean ± std
    """

    def __init__(self, csv_path: Optional[str] = None):
        self.records: list[dict] = []
        self.csv_path = Path(csv_path) if csv_path else None

    def timer(self, **meta):
        """Decorator factory that logs wall time with metadata.

        Args:
            **meta: Arbitrary key-value pairs stored alongside the
                    timing (e.g. seed=42, model="reg", dataset="kaust").
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - t0
                record = {
                    "function": func.__qualname__,
                    "wall_time_s": round(elapsed, 3),
                    **meta,
                }
                self.records.append(record)
                if elapsed < 60:
                    print(f"[timer] {func.__qualname__} ({meta}): {elapsed:.2f}s")
                else:
                    m, s = divmod(elapsed, 60)
                    print(f"[timer] {func.__qualname__} ({meta}): {int(m)}m {s:.1f}s")
                return result

            return wrapper

        return decorator

    def save(self, path: Optional[str] = None):
        """Write all records to CSV."""
        out = Path(path) if path else self.csv_path
        if out is None:
            raise ValueError("No csv_path specified")
        if not self.records:
            print("[timer] No records to save.")
            return

        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.records[0].keys())
        # Collect all keys across records in case metadata varies
        for r in self.records:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"[timer] Saved {len(self.records)} records to {out}")

    def summary(self):
        """Print mean ± std wall time grouped by (function, model)."""
        from collections import defaultdict

        groups = defaultdict(list)
        for r in self.records:
            key = (r.get("function", ""), r.get("model", ""))
            groups[key].append(r["wall_time_s"])

        print("\n[timer] Wall-time summary:")
        for (func, model), times in sorted(groups.items()):
            import numpy as np

            arr = np.array(times)
            label = f"{func}" + (f" [{model}]" if model else "")
            if len(arr) == 1:
                print(f"  {label}: {arr[0]:.2f}s")
            else:
                print(f"  {label}: {arr.mean():.2f} ± {arr.std():.2f}s  (n={len(arr)})")
