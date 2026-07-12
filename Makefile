.PHONY: install lint test test-installed test-1v1 test-gpu-ready smoke train evaluate tui

PYTHON ?= $(shell if [ -x venv/bin/python ]; then echo venv/bin/python; else command -v python3; fi)
RUFF ?= $(shell if $(PYTHON) -m ruff --version >/dev/null 2>&1; then echo "$(PYTHON) -m ruff"; elif python3 -m ruff --version >/dev/null 2>&1; then echo "python3 -m ruff"; else echo "$(PYTHON) -m ruff"; fi)
RUN_DIR ?= runs/colonist_1v1

install:
	$(PYTHON) -m pip install -e ".[dev,gym,colonist,tui]"

lint:
	$(RUFF) check catanatron/catanatron examples tests

test:
	$(PYTHON) -m pytest

test-installed:
	@PYTHON_BIN="$$(command -v "$(PYTHON)")"; \
	case "$$PYTHON_BIN" in /*) ;; *) PYTHON_BIN="$(CURDIR)/$$PYTHON_BIN" ;; esac; \
	cd /tmp && "$$PYTHON_BIN" -c "import catanatron; print(catanatron.__file__)"
	$(PYTHON) -m pytest tests/test_gpu_readiness.py -k generator_resumes_without_duplicate_games

test-1v1:
	$(PYTHON) -m pytest tests/test_colonist_1v1.py tests/test_colonist_1v1_training.py tests/test_colonist_1v1_gym_training.py

test-gpu-ready:
	$(PYTHON) -m pytest \
		tests/test_gpu_readiness.py \
		tests/test_experiment_backlog.py \
		tests/test_colonist_1v1_training.py \
		tests/test_colonist_1v1_gym_training.py \
		tests/machine_learning/test_dataset_v2.py \
		tests/machine_learning/test_leaf_evaluation.py \
		tests/machine_learning/test_mcts.py

smoke:
	$(PYTHON) examples/colonist_1v1_train.py --preset smoke --run-dir $(RUN_DIR) --skip-final-eval

train:
	$(PYTHON) examples/colonist_1v1_train.py --preset standard --run-dir $(RUN_DIR) --mixed-league --tensorboard

evaluate:
	$(PYTHON) examples/colonist_1v1_evaluate.py --agent L:$(RUN_DIR)/colonist_maskable_ppo.zip --benchmark --protocol full --gates --report $(RUN_DIR)/evaluation.json

tui:
	$(PYTHON) examples/colonist_1v1_tui.py --run-dir $(RUN_DIR)
