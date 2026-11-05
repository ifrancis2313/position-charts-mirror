"""Pure two-panel chart builder.

`build_position_chart` takes underlying OHLC price data, entry/exit event
markers, and a `Position`, and returns a `plotly.graph_objects.Figure`.

Top panel: underlying price (line or candlestick) with entry/exit markers.
Bottom panel: `position.pnl_curve()` -- the position's aggregated theoretical
value / P&L curve. For a single-leg position this is already a plain P&L
line (see `Position.pnl_curve`'s docstring) -- this module does not special-
case leg count anywhere; it only ever calls `position.pnl_curve()`.
"""

from __future__ import annotations

from typing import Literal, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from position_charts.models import MarketEvent, Position

PriceMode = Literal["line", "candlestick"]

# ---------------------------------------------------------------------------
# Color constants for price and P&L series.
# ---------------------------------------------------------------------------
_PRICE_COLOR_LIGHT = "#2a78d6"  # categorical slot 1, blue
_PRICE_COLOR_DARK = "#3987e5"
_PNL_COLOR_LIGHT = "#2a78d6"  # same slot -- top and bottom panels never appear as
_PNL_COLOR_DARK = "#3987e5"  # "competing series," so reusing slot 1 is correct, not lazy

# Entry/exit marker colors: deliberately NOT the categorical palette (an
# entry/exit marker is an event annotation, not a data series) and
# deliberately NOT a green/red "good/bad" pairing -- exiting a position is
# not inherently a loss, so coloring exits red would misrepresent it. Entry
# uses the reserved "good" status color (opening a position is a neutral-to-
# positive action); exit uses secondary ink (neutral). Shape (triangle-up vs
# triangle-down) is the primary identity channel; marker symbols also
# distinguish the events in grayscale or print.
_ENTRY_COLOR = "#0ca30c"  # status "good"
_EXIT_COLOR = "#52514e"  # secondary ink

_ZERO_LINE_LIGHT = "#c3c2b7"  # "baseline / axis" role
_ZERO_LINE_DARK = "#383835"
_SURFACE_LIGHT = "#fcfcfb"
_SURFACE_DARK = "#1a1a19"
_TEXT_LIGHT = "#0b0b0b"
_TEXT_DARK = "#ffffff"
_GRID_LIGHT = "#e1e0d9"  # hairline gridline
_GRID_DARK = "#2c2c2a"

_FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# ---------------------------------------------------------------------------
# Marker-decluttering constants (bug fix -- dense/choppy periods used to
# stack same-priced markers illegibly at a fixed 12px size with zero
# separation logic). Markers whose dates fall within `threshold_days` of a
# neighbor (see `_decluttered_positions`) get a small vertical offset,
# alternating above/below their true price, sized as a fraction of the
# panel's own price range -- proportional jitter, not an arbitrary pixel
# nudge, so it scales sensibly whether the underlying trades in single
# digits or the thousands. The true price is never lost: it's always
# carried in `customdata` and shown via hover, independent of the rendered
# (possibly jittered) y position.
# ---------------------------------------------------------------------------
_DENSE_TIME_FRACTION = 0.015  # markers within this fraction of the total date span count as "dense"
_MIN_DENSE_THRESHOLD_DAYS = 1  # always cluster same/adjacent-day markers, even on a short date range
_JITTER_PRICE_FRACTION = 0.015  # per-slot vertical offset, as a fraction of the panel's price range


def build_position_chart(
    underlying: pd.DataFrame,
    events: Sequence[MarketEvent],
    position: Position,
    *,
    price_mode: PriceMode = "line",
    dark_mode: bool = False,
    title: str | None = None,
) -> go.Figure:
    """Build the two-panel leg-first position chart.

    Args:
        underlying: date-indexed OHLC data. Requires a `close` column for
            `price_mode="line"`; requires `open`/`high`/`low`/`close` for
            `price_mode="candlestick"`.
        events: entry/exit `MarketEvent`s to mark on the top panel. May be
            empty (no markers drawn). Markers whose dates land close
            together (a choppy period with several trades in quick
            succession) are automatically given a small vertical offset so
            they stay legible -- see `_decluttered_positions`. The true
            price is always shown on hover regardless of any offset.
        position: the `Position` (one or more `Leg`s) whose aggregated
            `pnl_curve()` drives the bottom panel.
        price_mode: "line" (default) or "candlestick".
        dark_mode: selects the dark-surface colors.
        title: figure title; defaults to "{strategy} -- {position_id}".

    Returns:
        A two-row `plotly.graph_objects.Figure`, shared x-axis, ready to
        hand to `st.plotly_chart` or any other Plotly consumer. Hover mode
        is "closest" (per-point), not "x unified" -- unified hover merges
        every trace's info at a shared x-position into one tooltip, which
        would pair an entry/exit marker from one trade with an unrelated
        trade's marker whenever their dates happen to land close together.
        "closest" isolates each point's own hover data, always.
    """
    _validate_underlying(underlying, price_mode)

    price_color = _PRICE_COLOR_DARK if dark_mode else _PRICE_COLOR_LIGHT
    pnl_color = _PNL_COLOR_DARK if dark_mode else _PNL_COLOR_LIGHT
    zero_color = _ZERO_LINE_DARK if dark_mode else _ZERO_LINE_LIGHT
    surface = _SURFACE_DARK if dark_mode else _SURFACE_LIGHT
    text_color = _TEXT_DARK if dark_mode else _TEXT_LIGHT
    grid_color = _GRID_DARK if dark_mode else _GRID_LIGHT

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.62, 0.38],
        subplot_titles=(
            f"{position.underlying_symbol} price",
            "Position P&L (theoretical value)",
        ),
    )

    _add_price_trace(fig, underlying, price_mode, price_color, row=1)
    _add_event_markers(fig, events, surface, underlying, row=1)

    pnl = position.pnl_curve()
    fig.add_trace(
        go.Scatter(
            x=pnl.index,
            y=pnl.values,
            mode="lines",
            line=dict(color=pnl_color, width=2),
            name="Position P&L",
            showlegend=False,  # single series in this panel -- no legend box needed
            hovertemplate="%{x|%Y-%m-%d}<br>P&L: $%{y:,.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color=zero_color, width=1), row=2, col=1)

    fig.update_layout(
        title=dict(
            text=title or f"{position.strategy} -- {position.position_id}",
            y=0.97,
            x=0.01,
            xanchor="left",
            yanchor="top",
        ),
        paper_bgcolor=surface,
        plot_bgcolor=surface,
        font=dict(color=text_color, family=_FONT_FAMILY),
        # Legend sits below the whole two-row figure, not above it. The
        # previous placement (y=1.06, just above row 1) crowded the figure
        # title and row 1's own subplot_title annotation into the same
        # sliver of vertical space -- moving the legend below the plot area
        # entirely (negative y, in the reserved bottom margin) makes overlap
        # with the title structurally impossible regardless of exact pixel
        # dimensions: the title anchors near the top of the whole figure
        # (y=0.97, yanchor="top", default container reference) while the
        # legend anchors below the bottom axis (y<0, paper reference).
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
        hovermode="closest",
        margin=dict(t=70, b=90, l=60, r=30),
    )
    fig.update_xaxes(gridcolor=grid_color, showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor=grid_color, showgrid=True, zeroline=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=2, col=1)
    if price_mode == "candlestick":
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    return fig


def _validate_underlying(underlying: pd.DataFrame, price_mode: PriceMode) -> None:
    if underlying.empty:
        raise ValueError("underlying price data must not be empty")
    if price_mode == "candlestick":
        required = {"open", "high", "low", "close"}
        missing = required - set(underlying.columns)
        if missing:
            raise ValueError(
                f"price_mode='candlestick' requires columns {sorted(required)}, "
                f"missing {sorted(missing)}"
            )
    else:
        if "close" not in underlying.columns:
            raise ValueError("price_mode='line' requires a 'close' column")


def _add_price_trace(
    fig: go.Figure, underlying: pd.DataFrame, price_mode: PriceMode, color: str, row: int
) -> None:
    if price_mode == "candlestick":
        fig.add_trace(
            go.Candlestick(
                x=underlying.index,
                open=underlying["open"],
                high=underlying["high"],
                low=underlying["low"],
                close=underlying["close"],
                name="Price",
                showlegend=True,
            ),
            row=row,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=underlying.index,
                y=underlying["close"],
                mode="lines",
                line=dict(color=color, width=2),
                name="Price",
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
            ),
            row=row,
            col=1,
        )


def _decluttered_positions(
    events: Sequence[MarketEvent], underlying: pd.DataFrame
) -> dict[int, float]:
    """Return `{index into events: vertical y-offset}` for markers whose
    dates land close enough to a neighbor that they'd visually stack in a
    choppy trading period. Isolated markers (no neighbor within the
    threshold) are absent from the returned dict, i.e. get zero offset.

    Clustering: events are sorted by date; consecutive events (in that
    sorted order) within `threshold_days` of each other chain into one
    cluster. `threshold_days` is `max(total_date_span * _DENSE_TIME_
    FRACTION, _MIN_DENSE_THRESHOLD_DAYS)` -- always at least one day, so
    same/adjacent-day markers cluster even on a short date range, and scales
    up proportionally for long-running positions.

    Offsetting: within a cluster of size >= 2, each marker gets a
    alternating above/below offset (0, +1, -1, +2, -2, ... slots), each slot
    worth `price_range * _JITTER_PRICE_FRACTION`. A cluster of size 1 (no
    close neighbor) gets no offset at all -- this function changes nothing
    for the common case of well-separated entry/exit markers.
    """
    if len(events) < 2:
        return {}
    if "close" not in underlying.columns or underlying.empty:
        return {}

    date_index = pd.DatetimeIndex(underlying.index)
    total_span_days = (date_index.max() - date_index.min()).days
    threshold_days = max(total_span_days * _DENSE_TIME_FRACTION, _MIN_DENSE_THRESHOLD_DAYS)
    threshold = pd.Timedelta(days=threshold_days)

    price_range = float(underlying["close"].max() - underlying["close"].min())
    if price_range <= 0:
        return {}
    step = price_range * _JITTER_PRICE_FRACTION

    order = sorted(range(len(events)), key=lambda i: events[i].date)

    offsets: dict[int, float] = {}

    def _flush(cluster: list[int]) -> None:
        if len(cluster) < 2:
            return
        for slot, idx in enumerate(cluster):
            magnitude = (slot + 1) // 2
            sign = 1 if slot % 2 == 0 else -1
            offsets[idx] = sign * magnitude * step

    cluster: list[int] = []
    prev_date: pd.Timestamp | None = None
    for idx in order:
        current_date = pd.Timestamp(events[idx].date)
        if prev_date is not None and (current_date - prev_date) <= threshold:
            cluster.append(idx)
        else:
            _flush(cluster)
            cluster = [idx]
        prev_date = current_date
    _flush(cluster)

    return offsets


def _add_event_markers(
    fig: go.Figure,
    events: Sequence[MarketEvent],
    surface: str,
    underlying: pd.DataFrame,
    row: int,
) -> None:
    offsets = _decluttered_positions(events, underlying)

    entries = [(i, e) for i, e in enumerate(events) if e.kind == "entry"]
    exits = [(i, e) for i, e in enumerate(events) if e.kind == "exit"]

    if entries:
        idxs, evs = zip(*entries)
        true_prices = [e.price for e in evs]
        render_y = [p + offsets.get(i, 0.0) for i, p in zip(idxs, true_prices)]
        fig.add_trace(
            go.Scatter(
                x=[e.date for e in evs],
                y=render_y,
                customdata=true_prices,
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color=_ENTRY_COLOR,
                    line=dict(width=2, color=surface),
                ),
                name="Entry",
                text=[e.label or "" for e in evs],
                hovertemplate="Entry %{x|%Y-%m-%d}<br>$%{customdata:,.2f}<br>%{text}<extra></extra>",
            ),
            row=row,
            col=1,
        )
    if exits:
        idxs, evs = zip(*exits)
        true_prices = [e.price for e in evs]
        render_y = [p + offsets.get(i, 0.0) for i, p in zip(idxs, true_prices)]
        fig.add_trace(
            go.Scatter(
                x=[e.date for e in evs],
                y=render_y,
                customdata=true_prices,
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color=_EXIT_COLOR,
                    line=dict(width=2, color=surface),
                ),
                name="Exit",
                text=[e.label or "" for e in evs],
                hovertemplate="Exit %{x|%Y-%m-%d}<br>$%{customdata:,.2f}<br>%{text}<extra></extra>",
            ),
            row=row,
            col=1,
        )
