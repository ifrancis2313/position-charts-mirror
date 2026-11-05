"""Synthetic fixture data for tests: a single-leg case and a multi-leg case shaped
like `calendar_spread` (two legs, two different expiries, opposite sides).

The multi-leg fixture is built so the aggregation math is checkable by hand
(see the docstring on `calendar_spread_fixture` for the exact numbers). This
fixture deliberately exercises the case
where one leg's `theo_curve` ends before the position's other leg (the front-
month leg closes/expires while the back-month leg is still open), which is
exactly the scenario `Position.pnl_curve`'s forward-fill-then-sum handles.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from position_charts.models import Leg, MarketEvent, Position

_DATES = pd.to_datetime(
    ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
)


def _ohlc(closes: list[float]) -> pd.DataFrame:
    """Build a minimal-but-complete OHLC frame from a close series (open =
    prior close, high/low a small band around close/open) -- enough to
    exercise both price_mode="line" and price_mode="candlestick".
    """
    opens = [closes[0]] + closes[:-1]
    df = pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) + 0.5 for o, c in zip(opens, closes)],
            "low": [min(o, c) - 0.5 for o, c in zip(opens, closes)],
            "close": closes,
        },
        index=_DATES,
    )
    df.index.name = "date"
    return df


def single_leg_fixture() -> tuple[pd.DataFrame, list[MarketEvent], Position]:
    """A naked short put, one leg, five days.

    theo_curve: [5.00, 4.50, 4.00, 3.50, 3.00] (premium decaying toward
    expiry, as expected for the short holder). signed_multiplier = -1 * 1 *
    100 = -100, so pnl(t) = (theo(t) - 5.00) * -100:
        day0: 0.00, day1: 50.00, day2: 100.00, day3: 150.00, day4: 200.00
    """
    theo_curve = pd.Series([5.00, 4.50, 4.00, 3.50, 3.00], index=_DATES)
    leg = Leg(
        leg_id="short_put",
        instrument="put",
        side="short",
        quantity=1,
        entry_price=5.00,
        theo_curve=theo_curve,
        strike=4500.0,
        expiry=date(2026, 2, 20),
        right="P",
        delta=-0.30,
    )
    position = Position(
        position_id="single-leg-demo",
        strategy="short_put",
        underlying_symbol="SPX",
        legs=(leg,),
    )

    underlying = _ohlc([4550.0, 4560.0, 4545.0, 4570.0, 4580.0])
    events = [
        MarketEvent(date=date(2026, 1, 5), price=4550.0, kind="entry", label="Short 4500P"),
        MarketEvent(date=date(2026, 1, 9), price=4580.0, kind="exit", label="Closed 4500P"),
    ]
    return underlying, events, position


def calendar_spread_fixture() -> tuple[pd.DataFrame, list[MarketEvent], Position]:
    """A `calendar_spread`: short front-month call + long back-month call,
    same strike (4500), two different expiries, opposite sides.

    Front leg (short 4500C, expiry 2026-01-16) only has a 4-point theo_curve
    (day0..day3) -- it's modeled as closing/expiring one day before the
    position's other leg and before the fixture's last date, deliberately,
    to exercise the forward-fill-on-early-leg-closure path:
        theo_curve: [2.00, 1.80, 1.50, 1.00]
        signed_multiplier = -1 * 1 * 100 = -100
        pnl(t) = (theo(t) - 2.00) * -100:
            day0: 0, day1: 20, day2: 50, day3: 100
        -> forward-filled to day4: 100 (held flat, leg already closed)

    Back leg (long 4500C, expiry 2026-02-20) has the full 5-point theo_curve:
        theo_curve: [3.50, 3.60, 3.55, 3.80, 4.00]
        signed_multiplier = +1 * 1 * 100 = +100
        pnl(t) = (theo(t) - 3.50) * 100:
            day0: 0, day1: 10, day2: 5, day3: 30, day4: 50

    Expected aggregated `position.pnl_curve()` (front ffilled + back, summed):
        day0: 0 + 0   =   0.00
        day1: 20 + 10 =  30.00
        day2: 50 + 5  =  55.00
        day3: 100 + 30 = 130.00
        day4: 100 + 50 = 150.00  (front held flat past its own last date)
    """
    front_theo = pd.Series([2.00, 1.80, 1.50, 1.00], index=_DATES[:4])
    back_theo = pd.Series([3.50, 3.60, 3.55, 3.80, 4.00], index=_DATES)

    front_leg = Leg(
        leg_id="front_short_call",
        instrument="call",
        side="short",
        quantity=1,
        entry_price=2.00,
        theo_curve=front_theo,
        strike=4500.0,
        expiry=date(2026, 1, 16),
        right="C",
        delta=0.45,
    )
    back_leg = Leg(
        leg_id="back_long_call",
        instrument="call",
        side="long",
        quantity=1,
        entry_price=3.50,
        theo_curve=back_theo,
        strike=4500.0,
        expiry=date(2026, 2, 20),
        right="C",
        delta=0.55,
    )
    position = Position(
        position_id="calendar-spread-demo",
        strategy="calendar_spread",
        underlying_symbol="SPX",
        legs=(front_leg, back_leg),
    )

    underlying = _ohlc([4500.0, 4510.0, 4495.0, 4505.0, 4515.0])
    events = [
        MarketEvent(
            date=date(2026, 1, 5),
            price=4500.0,
            kind="entry",
            label="Short 4500C (Jan16) / Long 4500C (Feb20)",
        ),
        MarketEvent(
            date=date(2026, 1, 9),
            price=4515.0,
            kind="exit",
            label="Closed calendar_spread",
        ),
    ]
    return underlying, events, position


_DENSE_DATES = pd.to_datetime(
    [f"2026-03-{d:02d}" for d in range(2, 11)]  # 2026-03-02 .. 2026-03-10, 9 days
)


def dense_markers_fixture() -> tuple[pd.DataFrame, list[MarketEvent], Position]:
    """A single-leg position with a *choppy* run of entry/exit `MarketEvent`s
    -- several trades landing on the same or adjacent dates (2026-03-03 and
    2026-03-04), then one well-separated trade (entry 2026-03-07, exit
    2026-03-10) -- built to exercise `_decluttered_positions`'s marker-
    decluttering logic (bug fix: dense periods used to stack same-priced
    markers illegibly with zero separation).

    Events, in order: entry_a/exit_a/entry_b land on 2026-03-03 (0-day gaps),
    exit_b lands 2026-03-04 (1-day gap from entry_b) -- all four chain into
    one dense cluster under the `max(span * 0.015, 1 day)` threshold (this
    fixture's 8-day underlying span keeps the threshold pinned at its 1-day
    floor). entry_c (2026-03-07) and exit_c (2026-03-10) are each >=3 days
    from their nearest neighbor -- both must stay isolated, unjittered.
    """
    theo_curve = pd.Series(
        [5.00, 4.80, 4.60, 4.40, 4.20, 4.00, 3.80, 3.60, 3.40], index=_DENSE_DATES
    )
    leg = Leg(
        leg_id="short_put_dense",
        instrument="put",
        side="short",
        quantity=1,
        entry_price=5.00,
        theo_curve=theo_curve,
        strike=4500.0,
        expiry=date(2026, 4, 17),
        right="P",
        delta=-0.30,
    )
    position = Position(
        position_id="dense-demo",
        strategy="short_put",
        underlying_symbol="SPX",
        legs=(leg,),
    )

    closes = [4550.0, 4552.0, 4551.0, 4553.0, 4549.0, 4547.0, 4560.0, 4575.0, 4600.0]
    opens = [closes[0]] + closes[:-1]
    underlying = pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) + 0.5 for o, c in zip(opens, closes)],
            "low": [min(o, c) - 0.5 for o, c in zip(opens, closes)],
            "close": closes,
        },
        index=_DENSE_DATES,
    )
    underlying.index.name = "date"

    events = [
        MarketEvent(date=date(2026, 3, 3), price=4550.0, kind="entry", label="Trade A entry"),
        MarketEvent(date=date(2026, 3, 3), price=4551.0, kind="exit", label="Trade A exit"),
        MarketEvent(date=date(2026, 3, 3), price=4549.0, kind="entry", label="Trade B entry"),
        MarketEvent(date=date(2026, 3, 4), price=4552.0, kind="exit", label="Trade B exit"),
        MarketEvent(date=date(2026, 3, 7), price=4600.0, kind="entry", label="Trade C entry"),
        MarketEvent(date=date(2026, 3, 10), price=4610.0, kind="exit", label="Trade C exit"),
    ]
    return underlying, events, position
