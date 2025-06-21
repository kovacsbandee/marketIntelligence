# Data Pipeline Refactoring & Productionization Plan

## Overview

This document outlines the steps and best practices to refactor, debug, format, add logging, and test a Python data pipeline project to make it production-ready and maintainable.

---

## 1️⃣ Project Familiarization

- [ ] **Directory Structure:**  
  Document the key folders and files (use `tree` or list relevant files).

- [ ] **Dependency List:**  
  Include `requirements.txt` or `pyproject.toml`.

- [ ] **Module/Function Overview:**  
  Summarize the core modules and their responsibilities (especially ETL, adapters, database, and transform utilities).

- [ ] **Testing Status:**  
  Note existing test strategy (pytest/unittest/manual), and location of test files.

- [ ] **Run & Config Instructions:**  
  Document entry points and how to run the pipeline (CLI, scheduler, etc.), as well as configuration files (YAML, .env, etc.).

---

## 2️⃣ Code & Data Audit

- [ ] **Review ETL & Standardization:**  
  Check for consistency, reusability, and code duplication in data cleaning/transformation functions.

- [ ] **Examine API Adapter Methods:**  
  Ensure each API call, data load, and error handling are robust and DRY (Don’t Repeat Yourself).

- [ ] **Error Handling Assessment:**  
  Ensure all exceptions are caught and meaningful errors are provided/logged.

---

## 3️⃣ Refactoring & Abstraction

- [ ] **Unify Data Cleaning:**  
  - Centralize date parsing and null-handling (`NaT`, `NaN`, empty string).
  - Abstract repeated type conversions into utility functions.
  - Use consistent column naming conventions.

- [ ] **Abstract Database Inserts:**  
  - One function to insert pandas DataFrames into ORM classes, with deduplication and error logging.

- [ ] **Logging Improvements:**  
  - Use Python’s built-in `logging` module.
  - Log to a rotating file (e.g., `logs/pipeline.log`).
  - Set appropriate log levels: DEBUG, INFO, WARNING, ERROR.
  - Include contextual info in logs (timestamp, module, severity).

- [ ] **Exception Handling:**  
  - Standardize `try/except` blocks.
  - Log tracebacks to file, but present clear error messages to users/ops.
  - Add fallback behavior or retries where needed.

- [ ] **Code Formatting & Typing:**  
  - Apply [Black](https://black.readthedocs.io/en/stable/) or [ruff](https://docs.astral.sh/ruff/) for code style.
  - Use type hints throughout.
  - Update function/class docstrings for clarity and completeness.

---

## 4️⃣ Testing

- [ ] **Unit Tests:**  
  - Ensure every core function (especially transformations) has a unit test covering edge cases.

- [ ] **Integration Tests:**  
  - Add tests for API adapters with sample/mock data.
  - Test database insert and deduplication logic.

- [ ] **Validation:**  
  - Write tests to check output dataframes for type correctness, nulls, duplicate keys, etc.

- [ ] **Test Data:**  
  - Store synthetic or anonymized test data for repeatable runs.

---

## 5️⃣ Deployment & CI

- [ ] **Logging in Production:**  
  - Ensure log files are rotated and old logs are archived/deleted.
  - Document log locations and formats.

- [ ] **CI Pipeline:**  
  - If not present, set up GitHub Actions/GitLab CI/other to run tests and linters on push/PR.

- [ ] **Documentation:**  
  - Summarize the above in a `README.md` or `CONTRIBUTING.md` for new team members.
  - Document new utility functions and error handling conventions.

---

## 6️⃣ Pain Points & Ongoing Improvement

- [ ] Document current pain points (e.g. silent errors, slow debugging, inconsistent outputs).
- [ ] Prioritize improvements based on impact and frequency.

---

## Template Logging Configuration

<details>
<summary>Sample <code>logging.conf</code></summary>

```ini
[loggers]
keys=root

[handlers]
keys=rotatingHandler

[formatters]
keys=default

[logger_root]
level=INFO
handlers=rotatingHandler

[handler_rotatingHandler]
class=handlers.RotatingFileHandler
level=INFO
formatter=default
args=('logs/pipeline.log', 'a', 1000000, 5)

[formatter_default]
format=%(asctime)s %(levelname)s [%(module)s] %(message)s
datefmt=%Y-%m-%d %H:%M:%S
