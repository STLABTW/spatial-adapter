"""Unit tests for spatial_adapter.utils.timer.TimerLog."""

import csv
import time

import pytest

from spatial_adapter.utils.timer import TimerLog


# Basic decorator behaviour
def test_timer_records_wall_time():
    log = TimerLog()

    @log.timer(seed=0)
    def sleep_short():
        time.sleep(0.05)

    sleep_short()

    assert len(log.records) == 1
    assert log.records[0]["wall_time_s"] >= 0.04
    assert log.records[0]["seed"] == 0


def test_timer_preserves_return_value():
    log = TimerLog()

    @log.timer()
    def add(a, b):
        return a + b

    result = add(3, 4)
    assert result == 7


def test_timer_preserves_function_name():
    log = TimerLog()

    @log.timer()
    def my_func():
        pass

    my_func()
    assert "my_func" in log.records[0]["function"]


# Metadata
def test_metadata_stored_in_record():
    log = TimerLog()

    @log.timer(seed=42, model="reg", dataset="kaust")
    def dummy():
        pass

    dummy()

    r = log.records[0]
    assert r["seed"] == 42
    assert r["model"] == "reg"
    assert r["dataset"] == "kaust"


def test_multiple_calls_accumulate():
    log = TimerLog()

    for i in range(5):

        @log.timer(seed=i)
        def dummy():
            pass

        dummy()

    assert len(log.records) == 5
    assert [r["seed"] for r in log.records] == list(range(5))


# CSV save
def test_save_creates_csv(tmp_path):
    csv_path = tmp_path / "timings.csv"
    log = TimerLog(str(csv_path))

    @log.timer(seed=1, model="unreg")
    def dummy():
        pass

    dummy()
    log.save()

    assert csv_path.exists()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["seed"] == "1"
    assert rows[0]["model"] == "unreg"
    assert float(rows[0]["wall_time_s"]) >= 0


def test_save_with_explicit_path(tmp_path):
    log = TimerLog()

    @log.timer(seed=0)
    def dummy():
        pass

    dummy()

    out = tmp_path / "sub" / "out.csv"
    log.save(str(out))
    assert out.exists()


def test_save_multiple_seeds(tmp_path):
    csv_path = tmp_path / "multi.csv"
    log = TimerLog(str(csv_path))

    for s in range(3):

        @log.timer(seed=s, model="reg")
        def dummy():
            time.sleep(0.01)

        dummy()

    log.save()

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    seeds = [int(r["seed"]) for r in rows]
    assert seeds == [0, 1, 2]


def test_save_no_records_prints_message(tmp_path, capsys):
    log = TimerLog(str(tmp_path / "empty.csv"))
    log.save()
    assert "No records" in capsys.readouterr().out


def test_save_without_path_raises():
    log = TimerLog()

    @log.timer()
    def dummy():
        pass

    dummy()

    with pytest.raises(ValueError, match="No csv_path"):
        log.save()


# Heterogeneous metadata across records
def test_save_heterogeneous_metadata(tmp_path):
    csv_path = tmp_path / "hetero.csv"
    log = TimerLog(str(csv_path))

    @log.timer(seed=0, model="unreg")
    def run_unreg():
        pass

    @log.timer(seed=0, model="reg", tau1=1e2)
    def run_reg():
        pass

    run_unreg()
    run_reg()
    log.save()

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    # tau1 should appear as a column, empty for unreg
    assert rows[1]["tau1"] == "100.0"
    assert rows[0].get("tau1", "") == ""


# Summary
def test_summary_prints(capsys):
    log = TimerLog()

    for i in range(3):

        @log.timer(model="reg")
        def dummy():
            time.sleep(0.01)

        dummy()

    log.summary()
    output = capsys.readouterr().out
    assert "Wall-time summary" in output
    assert "reg" in output
    assert "n=3" in output


def test_timer_formats_minutes_when_elapsed_over_60s(capsys, monkeypatch):
    """Long-running calls log as 'Xm Y.Ys' instead of seconds."""
    from spatial_adapter.utils import timer as timer_mod

    log = TimerLog()
    # Fake time.perf_counter so elapsed = 125.4 seconds
    fake_times = iter([0.0, 125.4])
    monkeypatch.setattr(timer_mod.time, "perf_counter", lambda: next(fake_times))

    @log.timer(model="slow")
    def slow():
        return "ok"

    slow()
    output = capsys.readouterr().out
    assert "2m" in output  # 125s → 2m 5.4s
    assert log.records[0]["wall_time_s"] == pytest.approx(125.4, abs=0.1)


def test_summary_single_record_prints_without_std(capsys):
    """With n=1, summary prints the single value (no mean ± std formatting)."""
    log = TimerLog()

    @log.timer(model="solo")
    def once():
        pass

    once()
    log.summary()
    output = capsys.readouterr().out
    assert "solo" in output
    # n>1 branch prints 'n=<count>'; single-record branch must not
    assert "n=" not in output
