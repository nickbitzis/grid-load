PORT ?= 8856

.PHONY: install test lint format train train-quick backtest gate run docker-build docker-run screenshot clean

install:
	uv sync --python 3.12

test:
	uv run pytest -q

lint:
	uvx ruff check .
	uvx ruff format --check .

format:
	uvx ruff format .
	uvx ruff check . --fix

train:  ## four years, 365-day backtest, monthly refits: about 3 minutes, writes data/artifacts.json
	uv run gridload train

train-quick:  ## 820 days, 300-day backtest: about 20 seconds
	uv run gridload train --quick

backtest:
	uv run gridload backtest

gate:  ## what CI runs: fails unless the model beats same hour last week and coverage is 75% to 85%
	uv run gridload backtest --quick --gate

run:
	uv run gridload serve --port $(PORT)

docker-build:
	docker build -t grid-load .

docker-run:
	docker run --rm -p $(PORT):8856 grid-load

screenshot:  ## needs the server running on $(PORT) and Google Chrome
	uv run python scripts/screenshot.py --port $(PORT)

clean:
	rm -rf .pytest_cache .ruff_cache data
