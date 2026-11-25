"""date_calc: dates, times and the Italian calendar (spec V3 section 10, decisions D19 and D24).

Key rule: durations in hours and minutes are computed in UTC. Subtracting two datetimes with the same
tzinfo ignores the clock change (it would give 24 h also on the March night, which lasts 23).
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..envelope import Ctx
from ..errors import ToolError

Op = Callable[[dict[str, Any], Ctx], dict[str, Any]]
OPS: dict[str, Op] = {}

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{1,2}):(\d{2})(?::(\d{2}))?)?$")
_DMY = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?$")
_REMINDER = re.compile(r"^\s*(\d+)\s*(m|min|h|d|bd|w)\s*$")


def op(name: str) -> Callable[[Op], Op]:
    def deco(fn: Op) -> Op:
        OPS[name] = fn
        return fn

    return deco


def handle(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    return OPS[args["operation"]](args, ctx)


def utc_now() -> datetime:
    """Current time in UTC (replaceable in the tests)."""
    return datetime.now(UTC)


# ------------------------------------------------------------------ Italian calendar


def easter(year: int) -> date:
    """Gregorian Easter, anonymous algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month, day = divmod(h + ll - 7 * m + 114, 31)
    return date(year, month, day + 1)


@lru_cache(maxsize=512)
def holidays_of(year: int) -> tuple[tuple[date, str], ...]:
    e = easter(year)
    items = [
        (date(year, 1, 1), "New Year's Day"),
        (date(year, 1, 6), "Epiphany"),
        (e, "Easter Sunday"),
        (e + timedelta(days=1), "Easter Monday"),
        (date(year, 4, 25), "Liberation Day"),
        (date(year, 5, 1), "Labour Day"),
        (date(year, 6, 2), "Republic Day"),
        (date(year, 8, 15), "Assumption Day"),
        (date(year, 11, 1), "All Saints' Day"),
        (date(year, 12, 8), "Immaculate Conception"),
        (date(year, 12, 25), "Christmas Day"),
        (date(year, 12, 26), "St. Stephen's Day"),
    ]
    if year >= 2026:
        # Law of 8 October 2025 no. 151 (GU no. 236 of 10/10/2025): 4 October national holiday from 2026.
        items.append((date(year, 10, 4), "St. Francis of Assisi, patron saint of Italy"))
    merged: dict[date, str] = {}
    for d, name in sorted(items):
        merged[d] = f"{merged[d]} / {name}" if d in merged else name
    return tuple(sorted(merged.items()))


def holiday_name(d: date) -> str | None:
    for hd, name in holidays_of(d.year):
        if hd == d:
            return name
    return None


def is_business(d: date) -> bool:
    return d.weekday() < 5 and holiday_name(d) is None


def add_business_days(d: date, n: int) -> date:
    step = 1 if n >= 0 else -1
    left = abs(n)
    if left > 100_000:
        raise ToolError("LIMIT_EXCEEDED", "At most 100,000 business days.")
    while left:
        d += timedelta(days=step)
        if is_business(d):
            left -= 1
    return d


def add_months(d: Any, n: int) -> tuple[Any, bool]:
    """Adds months on the calendar; a day beyond the end of the month becomes the last one (with a warning)."""
    total = d.year * 12 + (d.month - 1) + n
    y, m = divmod(total, 12)
    if not 1 <= y <= 9999:
        raise ToolError("INVALID_ARGUMENT", "Date outside the range 1-9999.")
    last = calendar.monthrange(y, m + 1)[1]
    day = min(d.day, last)
    return d.replace(year=y, month=m + 1, day=day), day != d.day


def _year_warning(ctx: Ctx, year: int) -> None:
    if year < 2001:
        ctx.warn(
            "Holidays according to the current civil calendar (in force since 2001): before they were partly different."
        )


# ------------------------------------------------------------------ reading the dates


def _tz(name: str | None) -> ZoneInfo:
    name = name or "Europe/Rome"
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ToolError(
            "INVALID_ARGUMENT", f"Unknown time zone: {name!r}.", '"Europe/Rome"'
        ) from None


def parse_dt(text: Any, field: str = "date") -> tuple[datetime, bool]:
    """Returns (datetime without time zone, has_time). Accepts ISO and dd/mm/yyyy."""
    if not isinstance(text, str) or not text.strip():
        raise ToolError(
            "INVALID_ARGUMENT", f"'{field}' is needed as a date.", '"2026-11-23" or "23/11/2026"'
        )
    t = text.strip()
    m = _ISO.match(t)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        hh, mm, ss = m[4], m[5], m[6]
    else:
        m = _DMY.match(t)
        if not m:
            raise ToolError(
                "INVALID_ARGUMENT", f"Unreadable date in '{field}': {t!r}.", '"2026-11-23T10:30"'
            )
        d, mo, y = int(m[1]), int(m[2]), int(m[3])
        hh, mm, ss = m[4], m[5], m[6]
    try:
        out = datetime(y, mo, d, int(hh or 0), int(mm or 0), int(ss or 0))
    except ValueError as exc:
        raise ToolError(
            "INVALID_ARGUMENT", f"Non-existent date in '{field}': {t!r} ({exc}).", '"2026-02-28"'
        ) from None
    return out, hh is not None


def _date_arg(args: dict[str, Any], field: str, tz: ZoneInfo) -> date:
    if args.get(field):
        return parse_dt(args[field], field)[0].date()
    return utc_now().astimezone(tz).date()


def localize(naive: datetime, tz: ZoneInfo, ctx: Ctx) -> datetime:
    """Local time -> datetime with time zone. Non-existent time: forwards; repeated time: the first. With a warning."""
    a = naive.replace(tzinfo=tz, fold=0)
    b = naive.replace(tzinfo=tz, fold=1)
    if a.utcoffset() == b.utcoffset():
        return a
    back = a.astimezone(UTC).astimezone(tz)
    if back.replace(tzinfo=None) == naive:
        ctx.warn(
            f"Repeated time {naive:%Y-%m-%d %H:%M} (return to standard time): using the first, {a.isoformat(timespec='seconds')}."
        )
        return a
    ctx.warn(
        f"Non-existent time {naive:%Y-%m-%d %H:%M} (switch to summer time): moved to {back.isoformat(timespec='seconds')}."
    )
    return back


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _offset_text(dt: datetime) -> str:
    off = dt.utcoffset() or timedelta(0)
    sign = "+" if off >= timedelta(0) else "-"
    off = abs(off)
    h, rem = divmod(int(off.total_seconds()), 3600)
    return f"{sign}{h:02d}:{rem // 60:02d}"


def _num(v: float) -> int | float:
    return int(v) if float(v).is_integer() else round(v, 6)


# ------------------------------------------------------------------ operations


@op("now")
def _now(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    tz = _tz(args.get("timezone"))
    now = utc_now().astimezone(tz).replace(microsecond=0)
    dst = bool(now.dst())
    ctx.formatted = f"{WEEKDAY_NAMES[now.weekday()]} {now:%Y-%m-%d %H:%M} ({_offset_text(now)})"
    return {
        "datetime": iso(now),
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "weekday": WEEKDAYS[now.weekday()],
        "utc_offset": _offset_text(now),
        "is_dst": dst,
    }


def _months_between(birth: date, ref: date) -> int:
    n = (ref.year - birth.year) * 12 + (ref.month - birth.month)
    while n > 0 and add_months(birth, n)[0] > ref:
        n -= 1
    while add_months(birth, n + 1)[0] <= ref:
        n += 1
    return max(n, 0)


def _birthday_in(birth: date, year: int) -> date:
    if birth.month == 2 and birth.day == 29 and not calendar.isleap(year):
        return date(year, 2, 28)
    return birth.replace(year=year)


@op("age")
def _age(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    tz = _tz(args.get("timezone"))
    if not args.get("birth_date"):
        raise ToolError("INVALID_ARGUMENT", "'birth_date' is needed.", '"2019-06-12"')
    birth = parse_dt(args["birth_date"], "birth_date")[0].date()
    ref = _date_arg(args, "ref_date", tz)
    if ref < birth:
        raise ToolError("INVALID_ARGUMENT", "ref_date is before the birth date.")
    months_total = _months_between(birth, ref)
    anchor = add_months(birth, months_total)[0]
    years, months = divmod(months_total, 12)
    days = (ref - anchor).days
    bday_this = _birthday_in(birth, ref.year)
    passed = ref >= bday_this
    nxt = bday_this if ref <= bday_this else _birthday_in(birth, ref.year + 1)
    if birth.month == 2 and birth.day == 29:
        ctx.warn("Born on 29/02: in non-leap years the birthday is counted on 28/02.")
    total_days = (ref - birth).days
    ctx.step(f"From {birth:%Y-%m-%d} to {ref:%Y-%m-%d}: {months_total} full months.")
    ctx.step(
        f"{months_total} months = {years} years and {months} months; from {anchor:%Y-%m-%d} to {ref:%Y-%m-%d}: {days} days."
    )
    ctx.step(
        f"Check: {birth:%Y-%m-%d} + {years} years {months} months {days} days = {anchor + timedelta(days=days):%Y-%m-%d}."
    )
    ctx.formatted = f"{years} years, {months} months and {days} days"
    return {
        "years": years,
        "months": months,
        "days": days,
        "total_months": months_total,
        "total_weeks": total_days // 7,
        "total_days": total_days,
        "birthday_passed_this_year": passed,
        "next_birthday": nxt.isoformat(),
        "days_to_next_birthday": (nxt - ref).days,
    }


@op("weekday")
def _weekday(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    d = parse_dt(args.get("date"), "date")[0].date()
    name = holiday_name(d)
    _year_warning(ctx, d.year)
    iso_year, iso_week, _ = d.isocalendar()
    ctx.formatted = f"{d:%Y-%m-%d} is a {WEEKDAY_NAMES[d.weekday()]}" + (
        f" ({name})" if name else ""
    )
    return {
        "date": d.isoformat(),
        "weekday": WEEKDAYS[d.weekday()],
        "iso_week": iso_week,
        "iso_year": iso_year,
        "day_of_year": d.timetuple().tm_yday,
        "is_holiday": name is not None,
        "holiday_name": name,
        "is_business_day": is_business(d),
    }


def _business_between(a: date, b: date) -> int:
    """Business days d with a < d <= b (negative if b < a)."""
    if b < a:
        return -_business_between(b, a)
    if (b - a).days > 400_000:
        raise ToolError("LIMIT_EXCEEDED", "Interval too long.")
    return sum(1 for k in range(1, (b - a).days + 1) if is_business(a + timedelta(days=k)))


@op("diff")
def _diff(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    tz = _tz(args.get("timezone"))
    a, a_time = parse_dt(args.get("date"), "date")
    if not args.get("end_date"):
        raise ToolError("INVALID_ARGUMENT", "Serve 'end_date'.", '"2026-12-31"')
    b, b_time = parse_dt(args["end_date"], "end_date")
    days = (b.date() - a.date()).days
    la, lb = localize(a, tz, ctx), localize(b, tz, ctx)
    elapsed = lb.astimezone(UTC) - la.astimezone(UTC)
    hours = elapsed.total_seconds() / 3600
    biz = _business_between(a.date(), b.date())
    if (
        (a_time or b_time)
        and abs(hours - days * 24) > 1e-9
        and abs(hours - (b - a).total_seconds() / 3600) > 1e-9
    ):
        ctx.warn(
            f"There is a clock change between the two dates: {_num(hours)} real hours have passed."
        )
    w, r = divmod(abs(days), 7)
    ctx.step(f"Calendar days: {days}.")
    ctx.step(f"Real hours (computed in UTC): {_num(hours)}.")
    ctx.step(f"Business days d with {a.date():%Y-%m-%d} < d <= {b.date():%Y-%m-%d}: {biz}.")
    ctx.formatted = f"{days} days ({biz} business days)"
    return {
        "days": days,
        "weeks": w if days >= 0 else -w,
        "remaining_days": r if days >= 0 else -r,
        "business_days": biz,
        "elapsed_hours": _num(hours),
        "elapsed_minutes": _num(elapsed.total_seconds() / 60),
    }


def _calendar_add(naive: datetime, args: dict[str, Any], ctx: Ctx) -> datetime:
    years, months = int(args.get("years", 0) or 0), int(args.get("months", 0) or 0)
    weeks, days = int(args.get("weeks", 0) or 0), int(args.get("days", 0) or 0)
    out = naive
    if years or months:
        out, clamped = add_months(out, years * 12 + months)
        if clamped:
            ctx.warn(
                f"Day {naive.day} does not exist in the target month: using the last day, {out:%Y-%m-%d}."
            )
    if args.get("business_days"):
        out = datetime.combine(add_business_days(out.date(), days), out.time())
        out += timedelta(weeks=weeks)
    else:
        out += timedelta(weeks=weeks, days=days)
    return out


@op("add")
def _add(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    tz = _tz(args.get("timezone"))
    start, has_time = parse_dt(args.get("date"), "date")
    out = _calendar_add(start, args, ctx)
    hours, minutes = int(args.get("hours", 0) or 0), int(args.get("minutes", 0) or 0)
    if not (hours or minutes) and not has_time:
        ctx.formatted = f"{out:%Y-%m-%d} ({WEEKDAY_NAMES[out.weekday()]})"
        return {"date": out.date().isoformat(), "weekday": WEEKDAYS[out.weekday()]}
    local = localize(out, tz, ctx)
    if hours or minutes:
        real = (local.astimezone(UTC) + timedelta(hours=hours, minutes=minutes)).astimezone(tz)
        if real.utcoffset() != local.utcoffset():
            ctx.warn(
                "Hours and minutes are real time: with the clock change the local time shifts "
                f"({_offset_text(local)} -> {_offset_text(real)})."
            )
        local = real
    ctx.formatted = f"{local:%Y-%m-%d %H:%M} ({_offset_text(local)})"
    return {"date": iso(local), "weekday": WEEKDAYS[local.weekday()]}


def _weekday_index(name: Any) -> int:
    if name not in WEEKDAYS:
        raise ToolError(
            "INVALID_ARGUMENT", "Serve 'target_weekday' (monday ... sunday).", '"thursday"'
        )
    return WEEKDAYS.index(name)


@op("resolve_relative")
def _resolve(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    tz = _tz(args.get("timezone"))
    ref = _date_arg(args, "ref_date", tz)
    if not args.get("target_weekday"):
        days = int(args.get("days", 0) or 0)
        d = (
            add_business_days(ref, days)
            if args.get("business_days")
            else ref + timedelta(days=days)
        )
        ctx.step(
            f"{ref:%Y-%m-%d} + {days}{' business' if args.get('business_days') else ''} days = {d:%Y-%m-%d}."
        )
        ctx.formatted = f"{d:%Y-%m-%d} ({WEEKDAY_NAMES[d.weekday()]})"
        return {"date": d.isoformat(), "weekday": WEEKDAYS[d.weekday()]}
    wd = _weekday_index(args["target_weekday"])
    which = args.get("which", "next")
    nth = int(args.get("nth", 1) or 1)
    monday = ref - timedelta(days=ref.weekday())
    if which == "next":
        d = ref + timedelta(days=(wd - ref.weekday() - 1) % 7 + 1)
    elif which == "previous":
        d = ref - timedelta(days=(ref.weekday() - wd - 1) % 7 + 1)
    elif which == "this_week":
        d = monday + timedelta(days=wd)
    elif which == "next_week":
        d = monday + timedelta(days=7 + wd)
    else:
        if nth < 1:
            raise ToolError("INVALID_ARGUMENT", "nth must be >= 1.", "2")
        d = ref + timedelta(days=(wd - ref.weekday() - 1) % 7 + 1 + 7 * (nth - 1))
    name = holiday_name(d)
    if name:
        ctx.warn(f"{d:%Y-%m-%d} is a holiday: {name}.")
    ctx.formatted = f"{d:%Y-%m-%d} ({WEEKDAY_NAMES[d.weekday()]})"
    return {"date": d.isoformat(), "weekday": WEEKDAYS[d.weekday()]}


def _reminder(start: datetime, text: str, tz: ZoneInfo, ctx: Ctx) -> datetime:
    m = _REMINDER.match(str(text))
    if not m:
        raise ToolError(
            "INVALID_ARGUMENT", f"Invalid reminder: {text!r}.", '["48h", "2d", "3bd", "30m"]'
        )
    n, unit = int(m[1]), m[2]
    if unit in {"m", "min", "h"}:
        delta = timedelta(minutes=n) if unit != "h" else timedelta(hours=n)
        return (start.astimezone(UTC) - delta).astimezone(tz)
    naive = start.replace(tzinfo=None)
    if unit == "d":
        target = naive - timedelta(days=n)
    elif unit == "w":
        target = naive - timedelta(weeks=n)
    else:
        target = datetime.combine(add_business_days(naive.date(), -n), naive.time())
    return localize(target, tz, ctx)


@op("event_schedule")
def _event(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    tz = _tz(args.get("timezone"))
    naive, _ = parse_dt(args.get("date"), "date")
    start = localize(naive, tz, ctx)
    dur = int(args.get("duration_minutes", 60) or 0)
    end = (start.astimezone(UTC) + timedelta(minutes=dur)).astimezone(tz)
    reminders = [iso(_reminder(start, r, tz, ctx)) for r in args.get("reminders_before") or []]
    ctx.step(f"Start: {iso(start)} (time zone {tz.key}, {_offset_text(start)}).")
    ctx.step(f"End: start + {dur} real minutes = {iso(end)}.")
    for r_text, r_iso in zip(args.get("reminders_before") or [], reminders, strict=True):
        ctx.step(f"Reminder {r_text}: {r_iso}.")
    ctx.formatted = f"{start:%Y-%m-%d %H:%M}-{end:%H:%M} ({_offset_text(start)})"
    return {
        "start": iso(start),
        "end": iso(end),
        "reminders": reminders,
        "start_utc": iso(start.astimezone(UTC)),
    }


@op("invoice_due")
def _invoice(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    d = parse_dt(args.get("date"), "date")[0].date()
    if args.get("dffm_days") is None:
        raise ToolError("INVALID_ARGUMENT", "'dffm_days' is needed (e.g. 30, 60, 90).", "30")
    n = int(args["dffm_days"])
    plus = d + timedelta(days=n)
    due = date(plus.year, plus.month, calendar.monthrange(plus.year, plus.month)[1])
    ctx.step(f"Invoice date {d:%Y-%m-%d} + {n} days = {plus:%Y-%m-%d}.")
    ctx.step(f"End of month (DFFM, Italian invoice term): {due:%Y-%m-%d}.")
    if args.get("business_days") and not is_business(due):
        moved = due
        while not is_business(moved):
            moved += timedelta(days=1)
        ctx.step(f"{due:%Y-%m-%d} is not a business day: due date moved to {moved:%Y-%m-%d}.")
        ctx.warn(f"Due date moved from {due:%Y-%m-%d} to the first business day.")
        due = moved
    ctx.formatted = f"Due date: {due:%Y-%m-%d} ({WEEKDAY_NAMES[due.weekday()]})"
    return {"due_date": due.isoformat(), "weekday": WEEKDAYS[due.weekday()]}


def _year_month(text: Any) -> tuple[int, int]:
    if isinstance(text, str) and re.match(r"^\d{4}-\d{2}$", text.strip()):
        y, m = map(int, text.strip().split("-"))
        if not 1 <= m <= 12:
            raise ToolError("INVALID_ARGUMENT", f"Invalid month: {text!r}.", '"2026-12"')
        return y, m
    d = parse_dt(text, "date")[0]
    return d.year, d.month


@op("month_calendar")
def _month_calendar(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    y, m = _year_month(args.get("date"))
    weeks = [[day or None for day in week] for week in calendar.Calendar(0).monthdayscalendar(y, m)]
    hols = [{"date": d.isoformat(), "name": n} for d, n in holidays_of(y) if d.month == m]
    last = calendar.monthrange(y, m)[1]
    count = sum(1 for k in range(1, last + 1) if is_business(date(y, m, k)))
    _year_warning(ctx, y)
    ctx.formatted = f"{calendar.month_name[m]} {y}: {count} business days"
    return {"year": y, "month": m, "weeks": weeks, "holidays": hols, "business_days_count": count}


@op("holidays")
def _holidays(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    raw = args.get("date")
    if isinstance(raw, str) and re.match(r"^\d{4}$", raw.strip()):
        y = int(raw)
    elif raw:
        y = parse_dt(raw, "date")[0].year
    else:
        y = utc_now().astimezone(_tz(args.get("timezone"))).year
    if y < 1583:
        raise ToolError("INVALID_ARGUMENT", "Gregorian calendar: year from 1583 onwards.", '"2027"')
    _year_warning(ctx, y)
    e = easter(y)
    hols = [
        {"date": d.isoformat(), "name": n, "weekday": WEEKDAYS[d.weekday()]}
        for d, n in holidays_of(y)
    ]
    ctx.formatted = f"{len(hols)} holidays in {y}; Easter {e:%Y-%m-%d}"
    return {
        "year": y,
        "holidays": hols,
        "easter": e.isoformat(),
        "easter_monday": (e + timedelta(days=1)).isoformat(),
    }


def _nth_weekday(y: int, m: int, wd: int, k: int) -> date | None:
    days = [
        date(y, m, d)
        for d in range(1, calendar.monthrange(y, m)[1] + 1)
        if date(y, m, d).weekday() == wd
    ]
    if k == -1:
        return days[-1]
    return days[k - 1] if k <= len(days) else None


@op("recurring")
def _recurring(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    tz = _tz(args.get("timezone"))
    first, _ = parse_dt(args.get("date"), "date")
    count = args.get("count")
    end = parse_dt(args["end_date"], "end_date") if args.get("end_date") else None
    if count is None and end is None:
        raise ToolError("INVALID_ARGUMENT", "'count' or 'end_date' is needed.", '{"count": 10}')
    cap = int(count) if count is not None else 1000
    end_naive = None
    if end is not None:
        end_naive = end[0] if end[1] else datetime.combine(end[0].date(), datetime.max.time())
    out: list[datetime] = []
    clamped_any = False
    if args.get("target_weekday") or args.get("month_week"):
        wd = _weekday_index(args.get("target_weekday"))
        k = args.get("month_week")
        if k is None:
            raise ToolError(
                "INVALID_ARGUMENT",
                "With target_weekday 'month_week' is needed (1-5 or -1 = last).",
                "-1",
            )
        missing = 0
        i = 0
        while len(out) < cap and i < 12 * 1000:
            ym, _ = add_months(first.date().replace(day=1), i)
            i += 1
            d = _nth_weekday(ym.year, ym.month, wd, int(k))
            if d is None:
                missing += 1
                continue
            cand = datetime.combine(d, first.time())
            if cand < first:
                continue
            if end_naive is not None and cand > end_naive:
                break
            out.append(cand)
        if missing:
            ctx.warn(f"{missing} months without the {k}th {WEEKDAY_NAMES[wd]}: skipped.")
    else:
        units = {u: int(args.get(u, 0) or 0) for u in ("days", "weeks", "months", "years")}
        nonzero = {u: v for u, v in units.items() if v}
        if len(nonzero) != 1:
            raise ToolError(
                "INVALID_ARGUMENT",
                "Give only one non-zero interval among days, weeks, months, years.",
                '{"weeks": 1}',
            )
        unit, step = next(iter(nonzero.items()))
        if step < 0:
            raise ToolError("INVALID_ARGUMENT", "The interval must be positive.", '{"weeks": 1}')
        i = 0
        while len(out) < cap:
            if unit in {"months", "years"}:
                cand, clamped = add_months(first, i * step * (12 if unit == "years" else 1))
                clamped_any = clamped_any or clamped
            else:
                cand = first + timedelta(days=i * step * (7 if unit == "weeks" else 1))
            i += 1
            if end_naive is not None and cand > end_naive:
                break
            out.append(cand)
        if clamped_any:
            ctx.warn(
                f"Day {first.day} does not exist in some months: using the last day of the month."
            )
    if count is None and len(out) >= cap:
        ctx.warn("List stopped at 1000 occurrences.")
    occ = [iso(localize(o, tz, ctx)) for o in out]
    ctx.formatted = f"{len(occ)} occurrences" + (f", first {occ[0]}" if occ else "")
    return {"occurrences": occ, "count": len(occ)}


@op("convert_tz")
def _convert_tz(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    src_tz = _tz(args.get("timezone"))
    if not args.get("to_timezone"):
        raise ToolError(
            "INVALID_ARGUMENT", "'to_timezone' is needed (IANA time zone).", '"America/New_York"'
        )
    dst_tz = _tz(args["to_timezone"])
    naive, _ = parse_dt(args.get("date"), "date")
    src = localize(naive, src_tz, ctx)
    dst = src.astimezone(dst_tz)
    diff = (
        (dst.utcoffset() or timedelta(0)) - (src.utcoffset() or timedelta(0))
    ).total_seconds() / 3600
    ctx.step(f"{iso(src)} = {iso(src.astimezone(UTC))} in UTC.")
    ctx.step(f"In {dst_tz.key}: {iso(dst)}.")
    ctx.formatted = f"{src:%H:%M} {src_tz.key} = {dst:%H:%M} {dst_tz.key} ({dst:%Y-%m-%d})"
    return {"source": iso(src), "target": iso(dst), "offset_difference_hours": _num(diff)}
