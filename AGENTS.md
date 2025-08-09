# AGENTS Guidelines

## Overview
This project is intended to help me in having a data based understanding of stock market behaviour aiming for a profitable trading strategy. 
The project will evolve into an LLM based trading agent, with RAG architecuture applying Claude (if needed LLM fine tuning is required!)
Right now the data source is the alphavantage API the database for daily price data, company financials and other tabular information.
The data pipeline and the postgre db is up and running, dockerization and the initial load is needed.
The next phases and steps are the following:

    Phase 1. 
        - from PostgreDB to Dash app with news capabilities
        - An interface is needed from the database to the application with an intermediate optional data enrichment step 
          to calculate various price indicators and other derived information from the raw data. An input object will be the result of this step.
        - Creating an informative and logical visualization system with plotly plots for all the information stored in the database. 
          (big scale info panels with DMC's tab feature: 
              -> full scale oveview of the market, indices, somehow overall market state, calculation when the app is starting to run, but it is on the background
              -> P/E states, and blue chip's expected earnings report, symbol interrelations, graph based on the autocorrelations across industries, industry autocorrelations,
              -> individual stock base price chart, with company fundamentals, insider tradings and other stuffs always a random when the app is, started,
                  --> create the balance, cash flow, earnings, income tabs for the corresponding symbol and the price chart next to them)
        - An elasticsarch database will be created with interfaces to the marketIntelligence project to make it possible to store news data.
        - An interface between the API and the Elasticsearch DB is required!
        - Place these visualiuzations into a dash application with the capability to download company related news from alpha vantage API from a defined time period.

    Phase 2. - dash app usage and trend reversal analysis
        - applying Dash app to create ideas for mass statistics about trend reversals.
        - testing the derived trend reversal indicator in mass (on all the available stocks!)
        

## Directory Highlights
- `data_manager/`: ETL jobs, database builders
- `scripts/`: entry points for data loading
- `utils/`: shared helpers (e.g., logging)
- `visualization/`: plotting utilities
(etc.)

## Setup
1. `pip install -r requirements.txt`
2. Configure environment variables for database access (`DB_HOST`, `DB_NAME`, ...)

## Running the Pipeline
Describe how to use `scripts/run_initial_load.py` or other scripts.

## Coding Standards
- Use type hints and docstrings.
- Run `black`/`ruff` before committing.
- Reference the refactor plan for further guidance.

## Testing
Instructions on how to run current or future tests (e.g., `pytest`).

## Logging
Explain the log directory and how logs rotate.

## Contribution Workflow
Steps for opening pull requests, running tests, and peer review.
