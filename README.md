# Structured Deliberation for Financial Information Systems

This repository contains the code for a modular multi-agent research system that
produces sequential allocation decisions using specialised analyst, coordinator,
and risk-evaluation agents. The codebase is organised to separate the reproducible
runtime from archival exploration notebooks.

> This repository is anonymized for double-blind review.
> Identifying information will be added after publication.

## Repository Structure

```text
.
|-- analysis/     # Post-processing and metric aggregation utilities
|-- config/       # YAML configs for reasoning protocol variants
|-- scripts/      # Command-line tools for batch runs and reports
|-- src/          # Library code: agent graph, data loaders, orchestration
|-- tests/        # Unit tests for deterministic components
`-- pyproject.toml
```

## Reviewable Decision Traces

The system generates schema-validated JSON decision traces for every
deliberation cycle. These records capture rationales, confidence levels, risk
flags, and changes introduced during synthesis, supporting process review and
reproducibility.

## Requirements

- Python 3.11+
- PEP-517 compliant installer (`pip`, `uv`, `poetry`, etc.)
- Alpha Vantage API key for market data
- Gemini API key for the default LLM configuration

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

All dependencies are declared in `pyproject.toml`. CPU execution is supported.

## Running an Experiment

```bash
python -m src.main --config config/run_flat.yaml
```

Key arguments:

- `--config`: YAML specification for universe, data sources, and agent protocol
- `--regimes`: Optional regime definitions for segmented evaluation
- `--baseline-suite`: Execute baseline configurations sequentially

Outputs are saved to the directory specified in the YAML config.

The main experiments use Gemini through `GEMINI_API_KEY`. OpenAI dependencies
are retained for optional compatibility but are not required for the default
configuration. Full experiment runs require external API keys and may incur API
costs.

## Reproducing the Main Experiments

The main paper variants correspond to the following configuration files:

- Flat baseline: `config/run_flat.yaml`
- Voting baseline: `config/run_voting.yaml`
- Single-prompt baseline: `config/run_one_shot.yaml`
- MA-Reasoning full stack: `config/run_cot_debate_tom_sc.yaml`
- Schema ablations: `config/run_cot_sc_weights.yaml`, `config/run_cot_sc_off.yaml`

After running experiments, aggregate metrics with:

```bash
python analysis/aggregate_metrics.py runs/flat_2020_2024 runs/cot_debate_tom_sc_2020_2024
```

## Logging

Structured logging is provided through `structlog`.

Enable debug mode:

```bash
export DEBUG_MODE=1
```

## Tests

```bash
pytest
```

The test suite covers deterministic utilities, metrics, graph wiring, schema
validation, portfolio constraints, and reproducibility helpers.

## Data

To fetch fresh data:

```bash
export ALPHAVANTAGE_API_KEY=<key>
export GEMINI_API_KEY=<key>
```

Sample cached responses are included for offline testing where applicable.

## License

This repository is released under the MIT License for review and reproducibility
purposes. The author information is anonymized during double-blind review and
will be updated after publication.
