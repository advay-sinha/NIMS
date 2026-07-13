"""Tests for the offline Engine A ML workflow planner + CLI (no execution)."""

from __future__ import annotations

import pytest

from src.ml_workflow.planner import (
    DATASETS,
    MODELS,
    STEP_ORDER,
    build_plan,
    expand_datasets,
    expand_models,
    normalize_steps,
)


def test_expand_datasets_all_and_subset():
    assert expand_datasets(["all"]) == list(DATASETS)
    assert expand_datasets([]) == list(DATASETS)
    assert expand_datasets(["unsw_nb15"]) == ["unsw_nb15"]


def test_expand_datasets_rejects_unknown():
    with pytest.raises(ValueError):
        expand_datasets(["not_a_dataset"])


def test_expand_models_all_and_subset():
    assert expand_models(["all"]) == list(MODELS)
    assert expand_models(["xgboost", "lstm"]) == ["xgboost", "lstm"]


def test_normalize_steps_aliases_and_order():
    steps = normalize_steps(["train model", "validate dataset",
                             "resolve production model"])
    # Returned in canonical order regardless of input order.
    assert steps == ["validate", "train", "resolve"]


def test_normalize_steps_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_steps(["hack the firewall"])


def test_build_plan_single_dataset_single_model():
    plan = build_plan(["unsw_nb15"], ["xgboost"], ["all"])
    steps = [p.step for p in plan]
    # Canonical order preserved; global steps present exactly once.
    assert steps.index("validate") < steps.index("train") < steps.index("resolve")
    assert steps.count("validation_report") == 1
    train = next(p for p in plan if p.step == "train")
    assert train.args == ("--dataset", "unsw_nb15", "--model", "xgboost")
    assert train.display == \
        "python -m scripts.train_model --dataset unsw_nb15 --model xgboost"


def test_build_plan_scope_expansion():
    plan = build_plan(["nsl_kdd", "unsw_nb15"], ["xgboost", "lstm"],
                      ["preprocess", "train", "registry"])
    preprocess = [p for p in plan if p.step == "preprocess"]
    train = [p for p in plan if p.step == "train"]
    registry = [p for p in plan if p.step == "registry"]
    assert len(preprocess) == 2          # one per dataset
    assert len(train) == 4               # dataset x model
    assert len(registry) == 1            # global, once
    assert registry[0].args == ()


def test_build_plan_maps_to_real_scripts():
    plan = build_plan(["nsl_kdd"], ["xgboost"], ["all"])
    modules = {p.module for p in plan}
    for expected in ("scripts.validate_datasets", "scripts.train_model",
                     "scripts.build_model_registry", "scripts.resolve_model"):
        assert expected in modules


def test_all_steps_are_orderable():
    plan = build_plan(["nsl_kdd"], ["xgboost"], list(STEP_ORDER))
    order = [p.step for p in plan]
    # Each canonical step appears and validate precedes everything else.
    assert order[0] == "validate"
    assert set(STEP_ORDER).issubset(set(order))


# --------------------------------------------------------------- CLI


def test_cli_dry_run_prints_plan_and_runs_nothing(capsys, monkeypatch):
    import scripts.run_offline_ml_workflow as cli

    def _boom(step):
        raise AssertionError("must not execute during --dry-run")

    monkeypatch.setattr(cli, "_run_step", _boom)
    code = cli.main(["--dataset", "unsw_nb15", "--model", "xgboost", "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "python -m scripts.train_model --dataset unsw_nb15 --model xgboost" in out
    assert "nothing was executed" in out.lower()


def test_cli_list(capsys):
    import scripts.run_offline_ml_workflow as cli

    assert cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "unsw_nb15" in out and "xgboost" in out and "train" in out


def test_cli_rejects_unknown_dataset(capsys):
    import scripts.run_offline_ml_workflow as cli

    assert cli.main(["--dataset", "nope", "--model", "xgboost", "--dry-run"]) == 1


def test_cli_executes_via_run_step(monkeypatch):
    import scripts.run_offline_ml_workflow as cli

    calls: list = []
    monkeypatch.setattr(cli, "_run_step", lambda step: calls.append(step.module) or 0)
    code = cli.main(["--dataset", "nsl_kdd", "--model", "xgboost",
                     "--steps", "validate,registry"])
    assert code == 0
    assert calls == ["scripts.validate_datasets", "scripts.build_model_registry"]
