"""Tests for the leg-first Leg/Position model, independent of the chart layer.

The core behavior under test is `Position.pnl_curve`'s N-leg aggregation.
The `test_calendar_spread_*` tests check
exact per-date numbers (not just "doesn't crash" or "final value looks
plausible"), and `test_single_leg_degenerates_to_its_own_leg_curve` proves
the N=1 case isn't a separately-coded special case by asserting the
aggregate method and the raw leg method produce the identical series.
"""

from __future__ import annotations

import pandas as pd
import pytest

from position_charts.models import Leg, Position
from tests.fixtures import calendar_spread_fixture, single_leg_fixture


def test_single_leg_pnl_curve_matches_hand_computed_values():
    _underlying, _events, position = single_leg_fixture()
    pnl = position.pnl_curve()
    expected = [0.0, 50.0, 100.0, 150.0, 200.0]
    assert pnl.tolist() == pytest.approx(expected)


def test_single_leg_degenerates_to_its_own_leg_curve():
    """N=1 aggregation must be structurally identical to that one leg's own
    pnl_curve() -- not just numerically equal by coincidence, but produced by
    the exact same code path (Position.pnl_curve calls Leg.pnl_curve() for
    every leg and sums; with one leg, "sums" is a no-op).
    """
    _underlying, _events, position = single_leg_fixture()
    leg = position.legs[0]

    position_pnl = position.pnl_curve()
    leg_pnl = leg.pnl_curve()

    assert position_pnl.index.equals(leg_pnl.index)
    assert position_pnl.tolist() == pytest.approx(leg_pnl.tolist())


def test_calendar_spread_pnl_curve_matches_hand_computed_aggregation():
    """The core multi-leg test: two legs, different expiries, opposite sides
    (short front-month call, long back-month call), front leg's theo_curve
    ending before the back leg's. See calendar_spread_fixture's docstring
    for the full by-hand derivation of these five numbers.
    """
    _underlying, _events, position = calendar_spread_fixture()
    pnl = position.pnl_curve()
    expected = [0.0, 30.0, 55.0, 130.0, 150.0]
    assert pnl.tolist() == pytest.approx(expected)


def test_calendar_spread_front_leg_holds_flat_after_its_own_last_date():
    """Isolates the forward-fill behavior: the front leg's own pnl_curve has
    only 4 points (100.0 on its last date); the aggregated position curve's
    5th-date value must reflect that 100.0 held flat, not 0/NaN/dropped.
    """
    _underlying, _events, position = calendar_spread_fixture()
    front_leg, back_leg = position.legs

    front_pnl = front_leg.pnl_curve()
    back_pnl = back_leg.pnl_curve()
    assert len(front_pnl) == 4
    assert len(back_pnl) == 5
    assert front_pnl.iloc[-1] == pytest.approx(100.0)

    total = position.pnl_curve()
    assert len(total) == 5
    # last date: front held flat at 100.0 + back's own 5th value 50.0
    assert total.iloc[-1] == pytest.approx(100.0 + back_pnl.iloc[-1])


def test_calendar_spread_leg_signs_are_opposite():
    _underlying, _events, position = calendar_spread_fixture()
    front_leg, back_leg = position.legs
    assert front_leg.side == "short"
    assert back_leg.side == "long"
    assert front_leg.signed_multiplier() == -100
    assert back_leg.signed_multiplier() == 100
    # Same strike, different expiry, as in a calendar spread.
    assert front_leg.strike == back_leg.strike
    assert front_leg.expiry != back_leg.expiry


def _make_leg(**overrides) -> Leg:
    defaults = dict(
        leg_id="leg",
        instrument="call",
        side="long",
        quantity=1,
        entry_price=1.0,
        theo_curve=pd.Series([1.0, 1.5], index=pd.to_datetime(["2026-01-05", "2026-01-06"])),
    )
    defaults.update(overrides)
    return Leg(**defaults)


def test_leg_rejects_non_positive_quantity():
    with pytest.raises(ValueError, match="quantity"):
        _make_leg(quantity=0)


def test_leg_rejects_negative_entry_price():
    with pytest.raises(ValueError, match="entry_price"):
        _make_leg(entry_price=-1.0)


def test_leg_rejects_empty_theo_curve():
    with pytest.raises(ValueError, match="theo_curve"):
        _make_leg(theo_curve=pd.Series([], dtype=float))


def test_leg_rejects_unsorted_theo_curve_index():
    unsorted = pd.Series([1.0, 1.5], index=pd.to_datetime(["2026-01-06", "2026-01-05"]))
    with pytest.raises(ValueError, match="sorted"):
        _make_leg(theo_curve=unsorted)


def test_position_rejects_zero_legs():
    with pytest.raises(ValueError, match="at least one leg"):
        Position(position_id="p", strategy="s", underlying_symbol="SPX", legs=())


def test_position_rejects_duplicate_leg_ids():
    leg_a = _make_leg(leg_id="dup")
    leg_b = _make_leg(leg_id="dup")
    with pytest.raises(ValueError, match="duplicate"):
        Position(position_id="p", strategy="s", underlying_symbol="SPX", legs=(leg_a, leg_b))


def test_position_is_single_leg_property():
    _underlying, _events, single = single_leg_fixture()
    _underlying2, _events2, multi = calendar_spread_fixture()
    assert single.is_single_leg is True
    assert multi.is_single_leg is False
