"""Build a chart from synthetic price and option-value data."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from position_charts import Leg, MarketEvent, Position, build_position_chart


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        default="position-chart-example.html",
        help="HTML output path (default: %(default)s)",
    )
    args = parser.parse_args()

    dates = pd.date_range("2026-01-05", periods=5, freq="D")
    close = [100.0, 101.5, 99.5, 102.0, 103.0]
    theo_curve = pd.Series([2.0, 1.8, 1.5, 1.2, 1.0], index=dates)
    underlying = pd.DataFrame(
        {
            "open": [close[0], *close[:-1]],
            "high": [value + 0.5 for value in close],
            "low": [value - 0.5 for value in close],
            "close": close,
        },
        index=dates,
    )
    position = Position(
        position_id="synthetic-example",
        strategy="short_call",
        underlying_symbol="SYN",
        legs=(
            Leg(
                leg_id="short-call",
                instrument="call",
                side="short",
                quantity=1,
                entry_price=2.0,
                theo_curve=theo_curve,
            ),
        ),
    )
    events = [
        MarketEvent(date=date(2026, 1, 5), price=100.0, kind="entry", label="Synthetic entry"),
        MarketEvent(date=date(2026, 1, 9), price=103.0, kind="exit", label="Synthetic exit"),
    ]
    output = Path(args.output)
    build_position_chart(underlying, events, position).write_html(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
