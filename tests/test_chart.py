"""Tests for build_position_chart -- fully independent of any consuming app
(no Streamlit test harness; a plain plotly.graph_objects.Figure is
inspectable directly via fig.data / fig.layout).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import pytest

from position_charts.chart import build_position_chart
from position_charts.models import MarketEvent
from tests.fixtures import calendar_spread_fixture, dense_markers_fixture, single_leg_fixture


def _trace_by_name(fig: go.Figure, name: str) -> go.Scatter:
    matches = [t for t in fig.data if t.name == name]
    assert len(matches) == 1, f"expected exactly one trace named {name!r}, found {len(matches)}"
    return matches[0]


def test_single_leg_chart_has_expected_traces():
    underlying, events, position = single_leg_fixture()
    fig = build_position_chart(underlying, events, position)

    assert isinstance(fig, go.Figure)
    names = sorted(t.name for t in fig.data)
    # price line + entry marker + exit marker + P&L line
    assert names == ["Entry", "Exit", "Position P&L", "Price"]


def test_single_leg_chart_bottom_panel_matches_pnl_curve():
    underlying, events, position = single_leg_fixture()
    fig = build_position_chart(underlying, events, position)

    pnl_trace = _trace_by_name(fig, "Position P&L")
    expected = position.pnl_curve()
    assert list(pnl_trace.y) == pytest.approx(expected.tolist())


def test_single_leg_chart_degenerates_to_plain_pnl_line():
    """Bottom-panel trace for a single-leg position must equal that one
    leg's own pnl_curve() exactly -- same values the model-level test
    already proved equal position.pnl_curve(); this confirms the chart
    layer doesn't diverge from the model by, e.g., accidentally re-deriving
    P&L from the underlying price instead of from the leg's theo_curve.
    """
    underlying, events, position = single_leg_fixture()
    fig = build_position_chart(underlying, events, position)

    pnl_trace = _trace_by_name(fig, "Position P&L")
    leg_pnl = position.legs[0].pnl_curve()
    assert list(pnl_trace.y) == pytest.approx(leg_pnl.tolist())


def test_calendar_spread_chart_bottom_panel_matches_aggregated_curve():
    underlying, events, position = calendar_spread_fixture()
    fig = build_position_chart(underlying, events, position)

    pnl_trace = _trace_by_name(fig, "Position P&L")
    expected = [0.0, 30.0, 55.0, 130.0, 150.0]
    assert list(pnl_trace.y) == pytest.approx(expected)


def test_calendar_spread_chart_has_expected_traces():
    underlying, events, position = calendar_spread_fixture()
    fig = build_position_chart(underlying, events, position)
    names = sorted(t.name for t in fig.data)
    assert names == ["Entry", "Exit", "Position P&L", "Price"]


def test_entry_and_exit_markers_use_distinct_symbols_not_just_color():
    """dataviz rule: identity is never color-alone -- entry/exit must be
    distinguishable by marker symbol, independent of their colors.
    """
    underlying, events, position = single_leg_fixture()
    fig = build_position_chart(underlying, events, position)

    entry = _trace_by_name(fig, "Entry")
    exit_ = _trace_by_name(fig, "Exit")
    assert entry.marker.symbol == "triangle-up"
    assert exit_.marker.symbol == "triangle-down"
    assert entry.marker.symbol != exit_.marker.symbol
    assert entry.marker.color != exit_.marker.color


def test_no_events_yields_no_marker_traces():
    underlying, _events, position = single_leg_fixture()
    fig = build_position_chart(underlying, [], position)
    names = sorted(t.name for t in fig.data)
    assert names == ["Position P&L", "Price"]


def test_candlestick_mode_adds_candlestick_trace():
    underlying, events, position = single_leg_fixture()
    fig = build_position_chart(underlying, events, position, price_mode="candlestick")
    candle_traces = [t for t in fig.data if isinstance(t, go.Candlestick)]
    assert len(candle_traces) == 1


def test_candlestick_mode_requires_ohlc_columns():
    underlying, events, position = single_leg_fixture()
    line_only = underlying[["close"]]
    with pytest.raises(ValueError, match="candlestick"):
        build_position_chart(line_only, events, position, price_mode="candlestick")


def test_line_mode_requires_close_column():
    underlying, events, position = single_leg_fixture()
    no_close = underlying.drop(columns=["close"])
    with pytest.raises(ValueError, match="close"):
        build_position_chart(no_close, events, position, price_mode="line")


def test_empty_underlying_raises():
    _underlying, events, position = single_leg_fixture()
    empty = pd.DataFrame(columns=["close"])
    with pytest.raises(ValueError, match="empty"):
        build_position_chart(empty, events, position)


def test_dark_mode_changes_surface_color():
    underlying, events, position = single_leg_fixture()
    light_fig = build_position_chart(underlying, events, position, dark_mode=False)
    dark_fig = build_position_chart(underlying, events, position, dark_mode=True)
    assert light_fig.layout.paper_bgcolor != dark_fig.layout.paper_bgcolor


def test_figure_has_two_rows_with_shared_xaxis():
    underlying, events, position = single_leg_fixture()
    fig = build_position_chart(underlying, events, position)
    # subplot rows are addressed as xaxis/xaxis2, yaxis/yaxis2 by plotly;
    # shared_xaxes links the top row's axis to the bottom row's (xaxis.matches
    # == "x2"), not the reverse.
    assert fig.layout.xaxis2 is not None
    assert fig.layout.yaxis2 is not None
    assert fig.layout.xaxis.matches == "x2"


def test_custom_title_is_used():
    underlying, events, position = single_leg_fixture()
    fig = build_position_chart(underlying, events, position, title="My custom title")
    assert fig.layout.title.text == "My custom title"


def test_default_title_names_strategy_and_position_id():
    underlying, events, position = single_leg_fixture()
    fig = build_position_chart(underlying, events, position)
    assert position.strategy in fig.layout.title.text
    assert position.position_id in fig.layout.title.text


# ---------------------------------------------------------------------------
# Hover mode must keep each trace's hover info at a
# shared x-position into one tooltip -- an entry marker from one trade and
# an exit marker from a completely unrelated trade could appear paired in
# one tooltip if their dates happened to land close together. Fixed by
# switching to hovermode="closest" (per-point hover, isolated per trace).
# ---------------------------------------------------------------------------


def test_hovermode_is_closest_not_x_unified():
    """The load-bearing layout assertion for the cross-trade tooltip-pairing
    bug: "x unified" merges hover info across traces at a shared x; "closest"
    isolates each point's own hover data, which is the actual fix.
    """
    underlying, events, position = single_leg_fixture()
    fig = build_position_chart(underlying, events, position)
    assert fig.layout.hovermode == "closest"
    assert fig.layout.hovermode != "x unified"


def test_multi_trade_markers_sharing_a_date_keep_isolated_hover_text():
    """Synthetic multi-trade fixture: Trade A's exit and Trade B's entry both
    land on 2026-03-04 (one day after Trade A's entry, one day before Trade
    B's exit) -- exactly the "unrelated trades, close dates" shape the bug
    report describes. Confirms each marker still carries its own,
    un-merged label/price data (via `text`/`customdata`), and that the
    hovertemplate reads from that per-point data, not a shared/unified
    value -- this is what "closest" mode renders per marker.
    """
    underlying, _events, position = single_leg_fixture()
    events = [
        MarketEvent(date=date(2026, 1, 5), price=4550.0, kind="entry", label="Trade A entry"),
        MarketEvent(date=date(2026, 1, 6), price=4545.0, kind="exit", label="Trade A exit"),
        MarketEvent(date=date(2026, 1, 6), price=4560.0, kind="entry", label="Trade B entry"),
        MarketEvent(date=date(2026, 1, 9), price=4580.0, kind="exit", label="Trade B exit"),
    ]
    fig = build_position_chart(underlying, events, position)

    assert fig.layout.hovermode == "closest"

    entry_trace = _trace_by_name(fig, "Entry")
    exit_trace = _trace_by_name(fig, "Exit")

    # Trade A's exit (index 1, on 2026-01-06) and Trade B's entry (index 2,
    # also 2026-01-06) share a date -- each must still carry its own label,
    # not the other trade's.
    assert list(exit_trace.text) == ["Trade A exit", "Trade B exit"]
    assert list(entry_trace.text) == ["Trade A entry", "Trade B entry"]

    # hovertemplate renders from this trace's own %{text}/%{customdata} --
    # never a merged/shared value across traces, which is what "x unified"
    # would have produced at the shared 2026-01-06 x-position.
    assert "%{text}" in entry_trace.hovertemplate
    assert "%{text}" in exit_trace.hovertemplate
    assert list(entry_trace.customdata) == [4550.0, 4560.0]
    assert list(exit_trace.customdata) == [4545.0, 4580.0]


# ---------------------------------------------------------------------------
# Dense markers are decluttered rather than plotted at a fixed 12px
# size with zero decluttering -- markers close together in time (a choppy
# period with several trades) stacked illegibly. Fixed with a proportional
# vertical jitter for markers landing within a "dense" date threshold of
# each other; the true price is always preserved in `customdata`/hover.
# ---------------------------------------------------------------------------


def test_dense_markers_are_vertically_decluttered_without_losing_true_price():
    """`dense_markers_fixture` has 4 markers chained into one dense cluster
    (2026-03-03/2026-03-03/2026-03-03/2026-03-04) and 2 isolated markers
    (2026-03-07, 2026-03-10, each >=3 days from any neighbor). The naive
    baseline (pre-fix) renders every marker at its exact true price with no
    separation -- this asserts the fix actually moves clustered markers off
    their true price (a visible positioning change) while
    (a) never losing the true price (still exact in hover via customdata)
    and (b) leaving isolated markers completely untouched.
    """
    underlying, events, position = dense_markers_fixture()
    fig = build_position_chart(underlying, events, position)

    entry_trace = _trace_by_name(fig, "Entry")
    exit_trace = _trace_by_name(fig, "Exit")

    entry_render_y = list(entry_trace.y)
    entry_true_y = list(entry_trace.customdata)
    exit_render_y = list(exit_trace.y)
    exit_true_y = list(exit_trace.customdata)

    # True prices are always exactly recoverable via customdata, regardless
    # of any rendering offset -- hover never shows a fabricated price.
    assert entry_true_y == [4550.0, 4549.0, 4600.0]
    assert exit_true_y == [4551.0, 4552.0, 4610.0]

    # The dense cluster (entry_a, exit_a, entry_b, exit_b) must show real
    # separation vs. the naive baseline: at least one marker per trace is
    # rendered off its true price.
    assert entry_render_y != entry_true_y
    assert exit_render_y != exit_true_y
    # Both clustered exits (Trade A's and Trade B's, indices 0/1) get offset.
    assert exit_render_y[0] != pytest.approx(exit_true_y[0])
    assert exit_render_y[1] != pytest.approx(exit_true_y[1])
    # Trade B's clustered entry (index 1) gets offset.
    assert entry_render_y[1] != pytest.approx(entry_true_y[1])

    # Trade C's entry/exit are isolated (>=3 days from any neighbor) --
    # must render at their exact true price, completely unjittered.
    assert entry_render_y[2] == pytest.approx(entry_true_y[2])
    assert exit_render_y[2] == pytest.approx(exit_true_y[2])

    # Hover must read the true price from customdata, never the (possibly
    # offset) render position.
    assert "%{customdata" in entry_trace.hovertemplate
    assert "%{customdata" in exit_trace.hovertemplate


def test_sparse_markers_are_never_jittered():
    """Regression guard: existing well-separated fixtures (the common case)
    must render markers at their exact true price -- decluttering must never
    kick in when there's nothing dense to declutter.
    """
    underlying, events, position = single_leg_fixture()
    fig = build_position_chart(underlying, events, position)

    entry_trace = _trace_by_name(fig, "Entry")
    exit_trace = _trace_by_name(fig, "Exit")
    assert list(entry_trace.y) == [e.price for e in events if e.kind == "entry"]
    assert list(exit_trace.y) == [e.price for e in events if e.kind == "exit"]
    assert list(entry_trace.y) == list(entry_trace.customdata)
    assert list(exit_trace.y) == list(exit_trace.customdata)


# ---------------------------------------------------------------------------
# The legend position (y=1.06, just above row 1)
# crowded the figure title and row 1's own subplot_title annotation into the
# same sliver of vertical space. Fixed by moving the legend below the whole
# two-row figure (negative y, paper-relative, in the reserved bottom margin)
# and pinning the title near the very top of the container (y=0.97,
# yanchor="top") -- structurally disjoint regions of the canvas.
# ---------------------------------------------------------------------------


def test_legend_and_title_do_not_overlap():
    """Real numeric check, not a visual guess: the title is anchored within
    the top ~10% of the whole container (yanchor="top", y>=0.9, default
    "container" yref) while the legend is anchored below the entire plot
    area (y<0, its only valid reference is "paper", where 0 is the bottom
    of the plot) -- one lives at the very top of the canvas, the other
    below the bottom axis, so they cannot occupy the same space regardless
    of exact figure/margin pixel dimensions.
    """
    underlying, events, position = single_leg_fixture()
    fig = build_position_chart(underlying, events, position)

    title = fig.layout.title
    legend = fig.layout.legend

    assert title.yanchor == "top"
    assert title.yref in (None, "container")  # None resolves to plotly.js's "container" default
    assert title.y >= 0.9

    assert legend.y < 0  # strictly below the plot's own paper area (0 = bottom axis)
    assert legend.yanchor == "top"

    # Explicit separation margin so this doesn't just barely pass: title is
    # pinned within the top 10% of the container while legend sits at least
    # 15% below the bottom of the plot area -- comfortably disjoint.
    assert legend.y <= -0.15
