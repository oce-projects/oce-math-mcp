"""date_calc: all the cases of the spec (age, 29 February, Easter, daylight saving time, DFFM, relative)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from oce_math_mcp.tools import date_calc
from tests.conftest import call, err, ok


def dc(**kw: object) -> dict:
    return ok("date_calc", **kw)


EASTER = {
    2000: "2000-04-23",
    2019: "2019-04-21",
    2024: "2024-03-31",
    2025: "2025-04-20",
    2026: "2026-04-05",
    2027: "2027-03-28",
    2038: "2038-04-25",
    2285: "2285-03-22",
}


@pytest.mark.parametrize(("year", "easter"), EASTER.items())
def test_easter_known_dates(year: int, easter: str) -> None:
    r = dc(operation="holidays", date=str(year))
    assert r["easter"] == easter
    assert r["easter_monday"] == (date.fromisoformat(easter) + timedelta(days=1)).isoformat()


def test_easter_is_sunday_in_range_for_three_centuries() -> None:
    for year in range(1900, 2201):
        e = date.fromisoformat(date_calc.easter(year).isoformat())
        assert e.weekday() == 6
        assert date(year, 3, 22) <= e <= date(year, 4, 25)


def test_holidays_2026_include_st_francis() -> None:
    names = {h["date"]: h["name"] for h in dc(operation="holidays", date="2026")["holidays"]}
    assert "2026-10-04" in names
    assert len(names) == 13


def test_holidays_2025_without_st_francis() -> None:
    dates = [h["date"] for h in dc(operation="holidays", date="2025")["holidays"]]
    assert "2025-10-04" not in dates
    assert len(dates) == 12


def test_age_birthday_passed_and_not() -> None:
    r = dc(operation="age", birth_date="1990-05-15", ref_date="2026-11-23")
    assert r["years"] == 36 and r["birthday_passed_this_year"] is True
    assert r["total_days"] == (date(2026, 11, 23) - date(1990, 5, 15)).days
    r = dc(operation="age", birth_date="1990-05-15", ref_date="2026-05-14")
    assert r["years"] == 35 and r["birthday_passed_this_year"] is False
    assert r["days_to_next_birthday"] == 1


def test_age_born_on_february_29() -> None:
    resp = call("date_calc", operation="age", birth_date="2000-02-29", ref_date="2027-02-28")
    assert resp["result"]["years"] == 27
    assert resp["warnings"]
    assert dc(operation="age", birth_date="2000-02-29", ref_date="2027-02-27")["years"] == 26
    assert dc(operation="age", birth_date="2000-02-29", ref_date="2028-02-29")["years"] == 28


def test_age_months_and_days() -> None:
    r = dc(operation="age", birth_date="2000-01-15", ref_date="2000-03-20")
    assert (r["years"], r["months"], r["days"]) == (0, 2, 5)


def test_weekday_and_holiday_flags() -> None:
    r = dc(operation="weekday", date="2026-11-23")
    assert r["weekday"] == "monday" and "weekday_it" not in r and r["is_business_day"]
    r = dc(operation="weekday", date="2026-12-25")
    assert r["is_holiday"] and r["holiday_name"] == "Christmas Day" and not r["is_business_day"]
    r = dc(operation="weekday", date="2027-10-04")
    assert r["weekday"] == "monday" and r["is_holiday"] and not r["is_business_day"]


def test_day_month_year_format_accepted() -> None:
    assert dc(operation="weekday", date="23/11/2026")["weekday"] == "monday"


def test_invalid_date() -> None:
    assert err("date_calc", operation="weekday", date="2026-02-30") == "INVALID_ARGUMENT"


def test_diff_business_days_over_christmas() -> None:
    r = dc(operation="diff", date="2026-12-21", end_date="2026-12-31")
    assert r["days"] == 10 and r["business_days"] == 7


def test_add_months_end_of_month() -> None:
    assert dc(operation="add", date="2026-01-31", months=1)["date"] == "2026-02-28"
    assert dc(operation="add", date="2028-01-31", months=1)["date"] == "2028-02-29"


def test_add_business_days_skips_christmas() -> None:
    assert (
        dc(operation="add", date="2026-12-23", days=3, business_days=True)["date"] == "2026-12-29"
    )


@pytest.mark.parametrize(
    ("which", "nth", "expected"),
    [
        ("next", 1, "2026-11-26"),
        ("this_week", 1, "2026-11-26"),
        ("next_week", 1, "2026-12-03"),
        ("previous", 1, "2026-11-19"),
        ("nth_next", 2, "2026-12-03"),
    ],
)
def test_resolve_relative_thursday(which: str, nth: int, expected: str) -> None:
    r = dc(
        operation="resolve_relative",
        ref_date="2026-11-23",
        target_weekday="thursday",
        which=which,
        nth=nth,
    )
    assert r["date"] == expected


def test_resolve_relative_ten_business_days() -> None:
    r = dc(operation="resolve_relative", ref_date="2026-11-23", days=10, business_days=True)
    assert r["date"] == "2026-12-07"


def test_event_schedule_offsets_cet_cest() -> None:
    assert (
        dc(operation="event_schedule", date="2026-03-28T10:00")["start"]
        == "2026-03-28T10:00:00+01:00"
    )
    r = dc(operation="event_schedule", date="2026-03-29T10:00", duration_minutes=90)
    assert r["start"] == "2026-03-29T10:00:00+02:00"
    assert r["end"] == "2026-03-29T11:30:00+02:00"


def test_event_reminders_across_dst() -> None:
    r = dc(
        operation="event_schedule",
        date="2026-03-30T10:00",
        reminders_before=["48h", "2d", "3bd", "30m"],
    )
    assert r["reminders"] == [
        "2026-03-28T09:00:00+01:00",
        "2026-03-28T10:00:00+01:00",
        "2026-03-25T10:00:00+01:00",
        "2026-03-30T09:30:00+02:00",
    ]


def test_event_nonexistent_and_ambiguous_times() -> None:
    resp = call("date_calc", operation="event_schedule", date="2026-03-29T02:30")
    assert resp["result"]["start"] == "2026-03-29T03:30:00+02:00" and resp["warnings"]
    resp = call("date_calc", operation="event_schedule", date="2026-10-25T02:30")
    assert resp["result"]["start"] == "2026-10-25T02:30:00+02:00" and resp["warnings"]


@pytest.mark.parametrize(
    ("invoice", "days", "due"),
    [
        ("2026-01-15", 30, "2026-02-28"),
        ("2026-01-31", 60, "2026-04-30"),
        ("2026-11-30", 90, "2027-02-28"),
    ],
)
def test_invoice_due_dffm(invoice: str, days: int, due: str) -> None:
    assert dc(operation="invoice_due", date=invoice, dffm_days=days)["due_date"] == due


def test_invoice_due_moves_to_business_day() -> None:
    r = dc(operation="invoice_due", date="2026-05-01", dffm_days=30, business_days=True)
    assert r["due_date"] == "2026-06-01"


def test_month_calendar_december_2026() -> None:
    r = dc(operation="month_calendar", date="2026-12")
    assert r["business_days_count"] == 21
    assert {h["date"] for h in r["holidays"]} == {"2026-12-08", "2026-12-25", "2026-12-26"}
    assert all(len(w) == 7 for w in r["weeks"])


def test_now_uses_rome_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    fixed = datetime(2026, 11, 23, 9, 0, tzinfo=UTC)
    monkeypatch.setattr(date_calc, "utc_now", lambda: fixed)
    r = dc(operation="now")
    assert r["datetime"] == "2026-11-23T10:00:00+01:00"
    assert r["is_dst"] is False


# --- D24: age in months and weeks, real hours, recurrences, time zones ---------


def test_age_of_small_child_in_months_and_weeks() -> None:
    r = dc(operation="age", birth_date="2025-05-20", ref_date="2026-11-24")
    assert r["total_months"] == 18
    assert r["total_weeks"] == (date(2026, 11, 24) - date(2025, 5, 20)).days // 7


@pytest.mark.parametrize(
    ("start", "end", "hours"),
    [
        ("2026-03-28T12:00", "2026-03-29T12:00", 23),
        ("2026-10-24T12:00", "2026-10-25T12:00", 25),
        ("2026-11-23T12:00", "2026-11-24T12:00", 24),
    ],
)
def test_real_elapsed_hours_across_dst(start: str, end: str, hours: int) -> None:
    r = dc(operation="diff", date=start, end_date=end)
    assert r["elapsed_hours"] == hours
    assert r["elapsed_minutes"] == hours * 60


def test_add_hours_is_real_time_across_dst() -> None:
    resp = call("date_calc", operation="add", date="2026-03-28T23:30", hours=24)
    assert resp["result"]["date"] == "2026-03-30T00:30:00+02:00"
    assert resp["warnings"]


def test_recurring_weekly_keeps_local_time_across_dst() -> None:
    r = dc(operation="recurring", date="2026-03-24T10:00", weeks=1, count=2)
    assert r["occurrences"] == ["2026-03-24T10:00:00+01:00", "2026-03-31T10:00:00+02:00"]


def test_recurring_monthly_on_31st() -> None:
    resp = call("date_calc", operation="recurring", date="2026-01-31T09:00", months=1, count=4)
    assert [o[:10] for o in resp["result"]["occurrences"]] == [
        "2026-01-31",
        "2026-02-28",
        "2026-03-31",
        "2026-04-30",
    ]
    assert resp["warnings"]


def test_recurring_last_friday_of_month() -> None:
    r = dc(
        operation="recurring",
        date="2026-11-01T18:00",
        target_weekday="friday",
        month_week=-1,
        count=3,
    )
    assert [o[:10] for o in r["occurrences"]] == ["2026-11-27", "2026-12-25", "2027-01-29"]


def test_recurring_first_monday_until_end_date() -> None:
    r = dc(
        operation="recurring",
        date="2026-12-01T08:00",
        target_weekday="monday",
        month_week=1,
        end_date="2027-02-28",
    )
    assert [o[:10] for o in r["occurrences"]] == ["2026-12-07", "2027-01-04", "2027-02-01"]


def test_convert_tz_rome_to_new_york() -> None:
    r = dc(operation="convert_tz", date="2026-11-24T15:00", to_timezone="America/New_York")
    assert r["target"] == "2026-11-24T09:00:00-05:00"
    assert r["offset_difference_hours"] == -6
    r = dc(operation="convert_tz", date="2026-03-20T15:00", to_timezone="America/New_York")
    assert r["target"] == "2026-03-20T10:00:00-04:00"


def test_convert_tz_unknown_zone() -> None:
    assert (
        err("date_calc", operation="convert_tz", date="2026-11-24T15:00", to_timezone="Europe/Roma")
        == "INVALID_ARGUMENT"
    )
