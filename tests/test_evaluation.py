import csv
import json

import pytest

from reproduce import verify


def fixture_workload(tmp_path, monkeypatch, duplicate=False):
    monkeypatch.setattr("reproduce.ROOT", tmp_path)
    source = tmp_path / "queries.json"
    source.write_text(json.dumps({"parsed_plans": [{"plan_runtime": 1000}, {"plan_runtime": 2000}]}))
    manifest = {"train_databases": [str(i) for i in range(19)], "workloads": [
        {"role": "test", "path": "queries.json", "selected": 2}]}
    row = {"val_sample_idxs": [0, 0] if duplicate else [0, 1], "val_labels": [1., 2.],
           "val_preds": [2., 1.], "val_median_q_error_50": 2.,
           "val_median_q_error_95": 2., "val_median_q_error_100": 2.}
    with (tmp_path / "test_model_queries.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    return manifest


def test_q_error_and_runtime_units(tmp_path, monkeypatch):
    manifest = fixture_workload(tmp_path, monkeypatch)
    result = verify(tmp_path, manifest)["queries"]
    assert result == {"queries": 2, "median_q_error": 2., "p95_q_error": 2., "max_q_error": 2.}


def test_duplicate_predictions_are_rejected(tmp_path, monkeypatch):
    manifest = fixture_workload(tmp_path, monkeypatch, duplicate=True)
    with pytest.raises(AssertionError, match="Missing/duplicate"):
        verify(tmp_path, manifest)
