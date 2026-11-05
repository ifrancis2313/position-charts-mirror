# position-charts

## What it does

`position-charts` provides a leg-first Python model for positions and a Plotly chart builder that displays underlying prices, entry and exit markers, and aggregated theoretical P&L in two linked panels. Its inputs are pandas data frames, model objects, and market events; the chart builder returns a Plotly figure.

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/Ifrancis2313/position-charts-mirror.git
cd position-charts-mirror
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
pytest -q
python -m examples.basic_chart
```

The example uses generated data and writes `position-chart-example.html` to the current directory. Supply a path argument to choose a different output location.

## How it's tested

The test suite contains 32 tests covering model validation, single-leg and multi-leg P&L aggregation, chart traces, marker behavior, and layout. The aggregation tests assert exact hand-computed values: `[0, 50, 100, 150, 200]` for a single leg and `[0, 30, 55, 130, 150]` for a two-leg spread, including the flat contribution after one leg's time series ends.

## Design notes

- Positions are represented as tuples of one or more legs; P&L aggregation uses the same path for single-leg and multi-leg positions.
- Each leg's theoretical-value series can have its own date range. Aggregation aligns the series and carries the last known value forward after an earlier-ending series.
- The chart builder accepts line or candlestick price data and returns a Plotly figure without requiring a UI framework.
- Entry and exit events use distinct marker symbols; clustered markers are offset visually while their original prices remain available in hover data.

## Known limitations

- This package calculates P&L from supplied theoretical-value series; it does not source market data or estimate those values.
- A leg's contribution is carried forward after its last supplied date; callers are responsible for supplying series appropriate to their use case.
- Chart input validation checks required columns and non-empty price data, but does not validate every value or enforce date-index alignment.

## License

MIT. See [LICENSE](LICENSE).
