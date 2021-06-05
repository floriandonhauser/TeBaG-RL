SHELL=/bin/bash
LINT_PATHS=tests/ environments/ agents/
# inspired by stable baseline 3

pytest:
	./scripts/run_tests.sh

format:
	# Sort imports
	isort ${LINT_PATHS}
	# Reformat using black
	black -l 127 ${LINT_PATHS}

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://lintlyci.github.io/Flake8Rules/
	flake8 ${LINT_PATHS} --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings.
	flake8 ${LINT_PATHS} --max-line-length 120 --count --exit-zero --statistics

check-codestyle:
	# Sort imports
	isort --check ${LINT_PATHS}
	# Reformat using black
	black --check -l 127 ${LINT_PATHS}

commit-checks: format lint pytest

.PHONY: pytest format lint check-codestyle