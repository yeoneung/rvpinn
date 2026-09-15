PY ?= python

.PHONY: test verify-results verify-comparisons verify-evaluation audit-release smoke-exact analyze analyze-comparisons mechanism

test:
	$(PY) -m pytest tests -q

verify-results:
	$(PY) experiments/analyze_confirmatory.py

verify-comparisons:
	$(PY) experiments/verify_comparative_results.py

verify-evaluation:
	$(PY) experiments/verify_evaluation_results.py

audit-release:
	$(PY) experiments/audit_release_inputs.py --workspace .

smoke-exact:
	$(PY) experiments/smoke_exact_miqp.py --time-limit 10

analyze:
	$(PY) experiments/analyze_confirmatory.py
	$(PY) experiments/analyze_seasonal_sensitivity.py

analyze-comparisons:
	$(PY) experiments/analyze_confirmatory.py
	$(PY) experiments/analyze_comparative_reference.py
	$(PY) experiments/analyze_comparative_scenarios.py
	$(PY) experiments/make_comparative_parameter_tables.py

mechanism:
	$(PY) experiments/make_comparative_mechanism_figure.py --controller learned
	$(PY) experiments/make_comparative_mechanism_figure.py --controller reference
	$(PY) experiments/make_comparative_mechanism_figure.py --controller m2
	$(PY) experiments/make_comparative_mechanism_figure.py --controller m16
	$(PY) experiments/make_comparative_mechanism_figure.py --assemble
