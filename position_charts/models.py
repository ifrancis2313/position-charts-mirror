"""Leg-first position and leg models with count-agnostic P&L aggregation.

A position is a flat tuple of legs and may contain one or multiple legs.
Each leg has its own per-unit mark or theoretical-value time series.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

Instrument = Literal["call", "put", "future", "stock"]
Side = Literal["long", "short"]
Right = Literal["C", "P"]
EventKind = Literal["entry", "exit"]


@dataclass(frozen=True)
class Leg:
    """One leg of a (possibly multi-leg) position.

    `theo_curve` is a date-indexed, chronologically sorted series of this
    leg's own per-unit mark/theoretical price over its active life (i.e. from
    the position's entry date up to whichever comes first: this leg's own
    expiry, or the date it/the position was closed). It does NOT need to
    span the whole position's holding window -- a calendar spread's front-
    month leg legitimately has a shorter `theo_curve` than its back-month
    leg; see `Position.pnl_curve` for how that's handled.

    `quantity` is always positive; direction comes from `side`. `entry_price`
    is the premium/price paid or received per unit at entry, also always
    positive -- whether that was a debit or a credit is a function of `side`,
    not the sign of this field.
    """

    leg_id: str
    instrument: Instrument
    side: Side
    quantity: int
    entry_price: float
    theo_curve: pd.Series
    strike: float | None = None
    expiry: date | None = None
    right: Right | None = None  # redundant with instrument for options, kept for OTS-schema parity
    delta: float | None = None  # informational only (e.g. for hover tooltips) -- not used in aggregation
    multiplier: int = 100

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Leg {self.leg_id!r}: quantity must be positive, got {self.quantity}")
        if self.entry_price < 0:
            raise ValueError(f"Leg {self.leg_id!r}: entry_price must be non-negative, got {self.entry_price}")
        if self.multiplier <= 0:
            raise ValueError(f"Leg {self.leg_id!r}: multiplier must be positive, got {self.multiplier}")
        if self.theo_curve.empty:
            raise ValueError(f"Leg {self.leg_id!r}: theo_curve must not be empty")
        if not self.theo_curve.index.is_monotonic_increasing:
            raise ValueError(f"Leg {self.leg_id!r}: theo_curve index must be sorted ascending")

    def signed_multiplier(self) -> int:
        """+quantity*multiplier if long, -quantity*multiplier if short."""
        sign = 1 if self.side == "long" else -1
        return sign * self.quantity * self.multiplier

    def pnl_curve(self) -> pd.Series:
        """This leg's own theoretical P&L over its `theo_curve`'s date range:
        (mark - entry_price) * signed_multiplier, per unit time.
        """
        pnl = (self.theo_curve - self.entry_price) * self.signed_multiplier()
        pnl.name = self.leg_id
        return pnl


@dataclass(frozen=True)
class MarketEvent:
    """One entry or exit marker plotted on the top (underlying price) panel.

    `label` is optional free text shown in the marker's hover tooltip. The
    chart builder renders the supplied text without formatting it.
    """

    date: date
    price: float
    kind: EventKind
    label: str | None = None


@dataclass(frozen=True)
class Position:
    """A position is a flat, leg-first collection -- no named `short_put`/
    `long_call` fields, no assumption about leg count. `pnl_curve` is the
    single place N-leg aggregation happens; everything else in this package
    (the chart builder) only ever calls that method, never touches individual
    legs' curves directly, which is what makes single-leg degeneration
    automatic rather than a special case.
    """

    position_id: str
    strategy: str
    underlying_symbol: str
    legs: tuple[Leg, ...]

    def __post_init__(self) -> None:
        if len(self.legs) == 0:
            raise ValueError(f"Position {self.position_id!r}: must have at least one leg")
        leg_ids = [leg.leg_id for leg in self.legs]
        if len(leg_ids) != len(set(leg_ids)):
            raise ValueError(f"Position {self.position_id!r}: duplicate leg_id in {leg_ids}")

    @property
    def is_single_leg(self) -> bool:
        return len(self.legs) == 1

    def pnl_curve(self) -> pd.Series:
        """Aggregate theoretical P&L across every leg onto one shared,
        chronologically sorted date index.

        Algorithm (count-agnostic, N=1 included):
        1. Take the union of every leg's own `theo_curve` dates.
        2. Reindex each leg's `pnl_curve()` onto that union and forward-fill.
           Forward-filling means a leg that closes/expires before the
           position's other leg(s) holds its last known P&L flat afterward,
           rather than dropping out of the sum or discontinuously zeroing --
           e.g. a calendar spread's total P&L doesn't jump when the front
           leg expires while the back leg is still open.
        3. Sum leg-by-leg with `skipna=False`: if a leg's curve starts later
           than the union's earliest date (it wasn't open yet), the total is
           legitimately undefined (NaN) for those dates rather than silently
           treating a not-yet-opened leg as a zero contribution.

        For a single-leg position, the union index is exactly that one leg's
        own index, the reindex/ffill is a no-op, and the sum is over one
        term -- this *is* the plain P&L line, not a separate code path.
        """
        curves = [leg.pnl_curve() for leg in self.legs]
        union_index = curves[0].index
        for c in curves[1:]:
            union_index = union_index.union(c.index)
        union_index = union_index.sort_values()

        aligned = [c.reindex(union_index).ffill() for c in curves]
        total = pd.concat(aligned, axis=1).sum(axis=1, skipna=False)
        total.name = "position_pnl"
        return total
