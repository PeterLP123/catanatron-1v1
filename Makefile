.PHONY: install test test-1v1 smoke train evaluate tui

PYTHON ?= python3
RUN_DIR ?= runs/colonist_1v1

install:
	$(PYTHON) -m pip install -e ".[dev,gym,colonist,tui]"

test:
	$(PYTHON) -m pytest

test-1v1:
	$(PYTHON) -m pytest tests/test_colonist_1v1.py tests/test_colonist_1v1_training.py tests/test_colonist_1v1_gym_training.py

smoke:
	$(PYTHON) examples/colonist_1v1_train.py --preset smoke --run-dir $(RUN_DIR) --skip-final-eval

train:
	$(PYTHON) examples/colonist_1v1_train.py --preset standard --run-dir $(RUN_DIR) --mixed-league --tensorboard

evaluate:
	$(PYTHON) examples/colonist_1v1_evaluate.py --agent L:$(RUN_DIR)/colonist_maskable_ppo.zip --benchmark --protocol full --gates --report $(RUN_DIR)/evaluation.json

tui:
	$(PYTHON) examples/colonist_1v1_tui.py --run-dir $(RUN_DIR)
