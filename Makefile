PY ?= python

.PHONY: test verify-results verify-v4 verify-alignment audit-release audit-format smoke-exact analyze analyze-v4 mechanism materials manuscript

test:
	$(PY) -m pytest tests -q

verify-results:
	$(PY) experiments/analyze_confirmatory.py

verify-v4:
	$(PY) experiments/verify_round2_results.py

verify-alignment:
	$(PY) experiments/verify_alignment_results.py

audit-release:
	$(PY) experiments/audit_release_inputs.py --workspace .

audit-format:
	$(PY) experiments/audit_manuscript_format.py

smoke-exact:
	$(PY) experiments/smoke_exact_miqp.py --time-limit 10

analyze:
	$(PY) experiments/analyze_confirmatory.py
	$(PY) experiments/analyze_seasonal_sensitivity.py

analyze-v4:
	$(PY) experiments/analyze_confirmatory.py
	$(PY) experiments/analyze_round2_reference.py
	$(PY) experiments/analyze_round2_scenarios.py
	$(PY) experiments/make_round2_parameter_tables.py

materials:
	$(PY) experiments/make_submission_materials.py

mechanism:
	$(PY) experiments/make_round2_mechanism_figure.py --controller learned
	$(PY) experiments/make_round2_mechanism_figure.py --controller reference
	$(PY) experiments/make_round2_mechanism_figure.py --controller m2
	$(PY) experiments/make_round2_mechanism_figure.py --controller m16
	$(PY) experiments/make_round2_mechanism_figure.py --assemble

manuscript:
	$(PY) experiments/compile_manuscripts.py
