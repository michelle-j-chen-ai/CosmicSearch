"""The daily corpus refresh: when it wakes, and when it decides to do nothing.

The refresh itself (drop + reload of 34M rows) is not exercised here -- these
cover the decisions around it, which are what a bad night would get wrong.
"""

from __future__ import annotations

import calendar
import time

import pytest

import web_server as ws


def _utc_epoch(y, mo, d, h, mi, s=0) -> float:
    return float(calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0)))


class _Corpus:
    def __init__(self, version, rows=10, loaded_at=0.0):
        self.dataset_version = version
        self.num_rows = rows
        self.loaded_at = loaded_at


# --------------------------------- schedule ---------------------------------

@pytest.mark.parametrize(
    "now, expected_h",
    [
        (_utc_epoch(2026, 8, 24, 9, 0), 1.0),    # later today
        (_utc_epoch(2026, 8, 24, 10, 30), 23.5),  # already past: tomorrow
        (_utc_epoch(2026, 8, 24, 23, 59), 601 / 60),  # across midnight
    ],
)
def test_seconds_until_next_occurrence(now, expected_h):
    assert ws._seconds_until("10:00", now) == pytest.approx(expected_h * 3600, abs=1)


def test_seconds_until_is_never_zero_or_negative():
    """Exactly on the mark must mean tomorrow, not a zero sleep spinning the
    scheduler through repeated refreshes for a whole minute."""
    assert ws._seconds_until("10:00", _utc_epoch(2026, 8, 24, 10, 0)) == 86400


def test_consecutive_runs_are_24h_apart_across_a_dst_boundary():
    """UTC has no DST, so the gap is 24h even across a US transition."""
    before = _utc_epoch(2026, 11, 1, 5, 0)  # US DST ends 2026-11-01
    first = before + ws._seconds_until("10:00", before)
    second = first + ws._seconds_until("10:00", first)
    assert second - first == 86400


@pytest.mark.parametrize("bad", ["25:00", "10:61", "noon", "", "10"])
def test_invalid_times_are_rejected(bad):
    with pytest.raises(Exception):
        ws._seconds_until(bad, time.time())


def test_bad_config_disables_the_schedule_rather_than_crashing(monkeypatch):
    monkeypatch.setattr(ws, "_REFRESH_AT", "half past ten")
    started = []
    monkeypatch.setattr(ws.threading, "Thread", lambda **kw: started.append(kw))
    ws._start_corpus_refresh_schedule()
    assert not started
    assert "invalid" in ws._REFRESH["error"]


# ------------------------------ version gate --------------------------------

def test_unchanged_version_skips_the_reload(monkeypatch):
    resident = _Corpus(version=7)
    monkeypatch.setattr(ws, "_CORPORA", {ws.deployment.default(): {"status": "ready", "corpus": resident,
                                      "error": "", "started": 0.0}})
    monkeypatch.setitem(ws._REFRESH, "skipped", 0)
    _no_reload(monkeypatch)
    _table_at(monkeypatch, 7)
    assert "unchanged" in ws._corpus_refresh_tick()
    assert ws._REFRESH["skipped"] == 1


def test_moved_version_triggers_the_reload(monkeypatch):
    resident = _Corpus(version=7)
    full = {"status": "ready", "corpus": resident, "error": "", "started": 0.0}
    monkeypatch.setattr(ws, "_CORPORA", {ws.deployment.default(): full})
    _table_at(monkeypatch, 8)

    def _reload(project=None):
        full["corpus"] = _Corpus(version=8, rows=99)
    monkeypatch.setattr(ws, "_full_refresh_worker", _reload)

    assert ws._corpus_refresh_tick() == "reloaded 7 -> 8 (99 rows)"


def test_unknown_version_reloads_rather_than_assuming_sameness(monkeypatch):
    """A reader that reports no version is not evidence the table stood still."""
    full = {"status": "ready", "corpus": _Corpus(version=None), "error": "",
            "started": 0.0}
    monkeypatch.setattr(ws, "_CORPORA", {ws.deployment.default(): full})
    _table_at(monkeypatch, None)
    called = []
    monkeypatch.setattr(ws, "_full_refresh_worker",
                        lambda project=None: called.append(True) or full.update(
                            corpus=_Corpus(version=None, rows=5)))
    ws._corpus_refresh_tick()
    assert called


def test_no_resident_corpus_is_left_alone(monkeypatch):
    """Nothing is stale, and loading here would surprise an idle instance."""
    monkeypatch.setattr(ws, "_CORPORA", {ws.deployment.default(): {"status": "idle", "corpus": None,
                                      "error": "", "started": 0.0}})
    _no_reload(monkeypatch)
    assert ws._corpus_refresh_tick() == "skipped: no corpus resident"


def test_a_load_in_flight_is_not_restarted(monkeypatch):
    """Dropping a half-built corpus to start over is strictly worse than waiting."""
    monkeypatch.setattr(ws, "_CORPORA", {ws.deployment.default(): {"status": "loading", "corpus": None,
                                      "error": "", "started": 0.0}})
    _no_reload(monkeypatch)
    assert "already in flight" in ws._corpus_refresh_tick()


def test_a_failed_reload_is_reported_not_counted(monkeypatch):
    full = {"status": "error", "corpus": _Corpus(version=7), "error": "boom",
            "started": 0.0}
    monkeypatch.setattr(ws, "_CORPORA", {ws.deployment.default(): full})
    _table_at(monkeypatch, 8)
    monkeypatch.setattr(ws, "_full_refresh_worker",
                        lambda project=None: full.update(corpus=None))
    monkeypatch.setitem(ws._REFRESH, "refreshes", 0)
    assert ws._corpus_refresh_tick() == "failed: boom"
    assert ws._REFRESH["refreshes"] == 0


# --------------------------------- helpers ----------------------------------

def _table_at(monkeypatch, version):
    import full_corpus
    monkeypatch.setattr(full_corpus, "latest_version", lambda *a, **k: version)


def _no_reload(monkeypatch):
    def _boom(project=None):
        raise AssertionError("reloaded when it should not have")
    monkeypatch.setattr(ws, "_full_refresh_worker", _boom)


def test_the_daily_refresh_can_be_switched_off_by_name():
    """The schedule is disabled by an empty value, but the NLS_* vars are Secret
    Manager payloads so they survive a deploy -- and Secret Manager will not
    store an empty one. Without a word that means off there is no way to turn
    the schedule off durably."""
    import importlib
    import os

    import web_server

    try:
        for value in ("off", "OFF", " never ", "none", "disabled", ""):
            os.environ["NLS_CORPUS_REFRESH_UTC"] = value
            importlib.reload(web_server)
            assert web_server._REFRESH_AT == "", f"{value!r} should disable"
        for value, expected in (("10:00", "10:00"), (" 09:30 ", "09:30")):
            os.environ["NLS_CORPUS_REFRESH_UTC"] = value
            importlib.reload(web_server)
            assert web_server._REFRESH_AT == expected
    finally:
        os.environ.pop("NLS_CORPUS_REFRESH_UTC", None)
        importlib.reload(web_server)


def test_the_daily_refresh_is_off_unless_a_time_is_set():
    """Refreshing drops the resident corpus and rebuilds it, so it is a
    self-inflicted outage. Instances are replaced often enough on their own that
    a fresh process picks up the current table anyway."""
    import importlib
    import os

    import web_server

    os.environ.pop("NLS_CORPUS_REFRESH_UTC", None)
    importlib.reload(web_server)
    assert web_server._REFRESH_AT == ""


def test_a_corpus_that_cannot_load_says_so_out_loud(caplog, monkeypatch):
    """The credentials were revoked and the service 503'd for ten hours before a
    person noticed. Every instance fails identically, so nothing recovers on its
    own -- the only thing that shortens the outage is saying so."""
    import logging

    import oci_s3
    import web_server

    monkeypatch.setattr(web_server, "_ALERT_WEBHOOK", "")
    web_server._ALERTED.clear()
    with caplog.at_level(logging.ERROR, logger="nls"):
        web_server._alert_corpus_failed("neuron", oci_s3.CredentialsMissing("no AWS_* keys"))
    text = caplog.text
    assert "ALERT" in text and "neuron" in text
    # Names the cause, so whoever is paged knows it is not a transient blip.
    assert "credential" in text.lower()


def test_a_failing_alert_never_hides_the_failure(monkeypatch, caplog):
    """An unreachable webhook must not turn a corpus outage into a crash."""
    import logging

    import web_server

    monkeypatch.setattr(web_server, "_ALERT_WEBHOOK", "http://127.0.0.1:1/nope")
    web_server._ALERTED.clear()
    with caplog.at_level(logging.ERROR):
        web_server._alert_corpus_failed("frontier", RuntimeError("disk on fire"))
    assert "ALERT" in caplog.text and "disk on fire" in caplog.text
