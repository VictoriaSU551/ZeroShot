"""Independently audit every completed local ZeroShot short run."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from download_data import ROOT, sha256
from reproduce import verify, write_json


def main():
    summaries = []
    for summary_path in sorted((ROOT / "output").glob("smoke_*/summary.json")):
        run = summary_path.parent
        summary = json.loads(summary_path.read_text())
        binding = json.loads((run / "binding.json").read_text())
        command = json.loads((run / "command.json").read_text())
        inputs = Path(command[command.index("--statistics_file") + 1]).parent
        manifest = json.loads((inputs / "manifest.json").read_text())
        assert sha256(inputs / "manifest.json") == binding["inputs_sha256"]
        assert sha256(inputs / "statistics.json") == manifest["statistics_sha256"]
        for item in manifest["workloads"]:
            assert sha256(ROOT / item["path"]) == item["sha256"]
            assert sha256(ROOT / item["source"]) == item["source_sha256"]
        for name, digest in binding["code"].items():
            assert sha256(ROOT / name) == digest, f"Model code changed: {name}"
        assert sha256(ROOT / "train.py") == binding["train_py_sha256"]
        config = Path(command[command.index("--hyperparameter_path") + 1])
        assert sha256(config) == binding["config_sha256"]
        assert sha256(run / "model.pt") == summary["checkpoint_sha256"]
        receipt_path = run / "evaluation_order_fix.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            assert sha256(run / "model.pt") == receipt["checkpoint_sha256_before_reevaluation"]
            old_binding = json.loads((run / "training_binding.json").read_text())
            assert old_binding["code"][receipt["file"]] == receipt["before_sha256"]
            old_binding["code"][receipt["file"]] = receipt["after_sha256"]
            assert old_binding == binding, "Unexpected change beyond evaluation ordering"
        checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] + 1 == summary["epochs_completed"]
        assert checkpoint["finished"]
        assert all(torch.isfinite(tensor).all() for tensor in checkpoint["model"].values())
        with (run / "model.csv").open() as stream:
            epochs = list(csv.DictReader(stream))
        assert len(epochs) == summary["epochs_completed"]
        assert np.isfinite([float(e["mean_loss"]) for e in epochs]).all()
        assert verify(run, manifest) == summary["test_results"]
        assert summary["checkpoint_reload_matches"]
        assert "Successfully loaded checkpoint" in (run / "reload.log").read_text(encoding="utf-8")
        summaries.append(summary)
    if not summaries:
        raise RuntimeError("No completed runs found")
    write_json(ROOT / "output" / "reproduction_summary.json", {
        "scope": "Zero-Shot Cost Models, VLDB 2022 only",
        "upstream_commit": "9ceabbdd753b6ce0c21a23cd6e64214e60e1197c",
        "full_paper_reproduction": False,
        "runs": summaries,
    })
    write_json(ROOT / "output" / "verification.json", {
        "status": "PASS", "completed_runs": len(summaries),
        "checks": ["source and sampled input hashes", "feature statistics hash", "model code hashes",
                   "checkpoint hash and finite weights", "completed epochs and finite losses",
                   "complete test coverage", "positive finite predictions", "labels in seconds",
                   "independent Q-error calculation", "checkpoint reload receipt"],
    })
    print(f"PASS: {len(summaries)} completed runs; full paper reproduction remains incomplete.")


if __name__ == "__main__":
    main()
