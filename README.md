# Trade Execution System

An automated trading system that integrates with multiple cryptocurrency exchanges (OKX and Binance) to execute trading strategies across different timeframes.
YQ: OKX API interation, you could trade after configure your own OKX's key.
## Features

- Multi-exchange API integration (OKX v5, Binance)
- Multi-timeframe analysis capabilities
- Technical indicator calculations
- Strategy execution and visualization
- Database-backed data persistence

## Repository Structure

```
.
├── YQ/                 # Trading API integrations
├── strategy/           # Trading strategies
├── indicators/         # Technical indicators
├── tests/              # Unit tests
├── data/               # Market data storage
├── plots/              # Visualization outputs
└── *.py                # Main executable scripts
```

## Getting Started

1. Install Python 3.7+
2. Configure API credentials in \`Account.json\`
3. Run scripts directly: \`python3 <script_name>.py\`

## License

MIT
