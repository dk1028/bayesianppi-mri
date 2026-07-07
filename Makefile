PYTHON ?= python3
FULL_CSV ?= results/autorater/autorater_predictions_all_subject_oof_with_scan_age.csv
AGE6570_CSV ?= results/autorater/autorater_predictions_6570_subject_oof_scan_age.csv

.PHONY: setup test sbc-smoke sbc coverage case-study kbin all-analysis

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

test:
	pytest -q

sbc-smoke:
	$(PYTHON) code/ppi/sbc.py --output-dir /tmp/bayesianppi-sbc-smoke --M 20 --S 100 --NA 100 --NH 20 --priors uniform --seed 1

sbc:
	$(PYTHON) code/ppi/sbc.py --output-dir results/sbc --M 500 --S 1000 --NA 2116 --NH 100 --priors jeffreys uniform --bins 20 --seed 2025

coverage:
	$(PYTHON) code/ppi/coverage_experiment.py --full-csv $(FULL_CSV) --age6570-csv $(AGE6570_CSV) --output-dir results/coverage --nsim 500 --posterior-draws 5000 --bootstrap 1000 --seed 2025

case-study:
	$(PYTHON) code/ppi/case_study_analysis.py --input-csv $(FULL_CSV) --output-dir results/case_study --bootstrap 2000 --permutations 5000 --threshold-bootstrap 2000 --seed 2025

kbin:
	$(PYTHON) code/ppi/k_chain_rule.py --full-csv $(FULL_CSV) --age6570-csv $(AGE6570_CSV) --output-dir results/kbin --k 2 4 5 --draws 10000 --seed 2025

all-analysis: coverage case-study kbin sbc
