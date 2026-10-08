"""Reproducible short cross-database training using the authors' model."""
import argparse
import ast
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from download_data import ROOT, sha256

TRAIN_DBS = ["airline", "ssb", "tpc_h", "walmart", "financial", "basketball", "accidents",
             "movielens", "baseball", "hepatitis", "tournament", "credit", "employee",
             "consumer", "geneea", "genome", "carcinogenesis", "seznam", "fhnk"]
TRAIN_WORKLOADS = ["index_workload_100k_s2_c8220.json", "workload_100k_s1_c8220.json"]
TEST_WORKLOADS = TRAIN_WORKLOADS + ["synthetic_c8220.json", "scale_c8220.json", "job-light_c8220.json"]
CONFIGS = {"postgres": "tune_est_best_config.json", "deepdb": "tune_deepdb_best_config.json",
           "exact": "tune_best_config.json"}


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(samples, test_samples, seed):
    from models.preprocessing.feature_statistics import gather_feature_statistics
    target = ROOT / "output" / f"inputs_s{seed}_n{samples}_t{test_samples}"
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["workloads"]:
            if sha256(ROOT / item["source"]) != item["source_sha256"] or sha256(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("Input changed; use a fresh output directory")
        if sha256(target / "statistics.json") != manifest["statistics_sha256"]:
            raise ValueError("Feature statistics changed")
        return target, manifest
    rng = random.Random(seed)
    items = []
    for database in TRAIN_DBS + ["imdb"]:
        role = "test" if database == "imdb" else "train_validation"
        for workload in TEST_WORKLOADS if role == "test" else TRAIN_WORKLOADS:
            source = ROOT / "data" / "runs" / "deepdb_augmented" / database / workload
            content = json.loads(source.read_text(encoding="utf-8"))
            total = len(content["parsed_plans"])
            cap = test_samples if role == "test" else samples
            indices = sorted(rng.sample(range(total), min(cap, total))) if cap else list(range(total))
            content["parsed_plans"] = [content["parsed_plans"][i] for i in indices]
            dest = target / database / workload
            write_json(dest, content)
            items.append({"database": database, "role": role, "source": source.relative_to(ROOT).as_posix(),
                          "source_sha256": sha256(source), "path": dest.relative_to(ROOT).as_posix(),
                          "sha256": sha256(dest), "available": total, "selected": len(indices), "indices": indices})
    train_paths = [str(ROOT / i["path"]) for i in items if i["role"] == "train_validation"]
    # Fit metadata on source databases only. IMDB runtimes never enter training/validation.
    gather_feature_statistics(train_paths, str(target / "statistics.json"))
    manifest = {"seed": seed, "train_databases": TRAIN_DBS, "test_database": "imdb", "workloads": items,
                "statistics_scope": "training databases only (including their validation split)",
                "statistics_sha256": sha256(target / "statistics.json")}
    write_json(manifest_path, manifest)
    return target, manifest


def read_predictions(path):
    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    records = sorted(zip(ast.literal_eval(row["val_sample_idxs"]), ast.literal_eval(row["val_labels"]),
                         ast.literal_eval(row["val_preds"])))
    return row, records


def verify(run, manifest, reload_dir=None):
    import numpy as np
    results = {}
    tests = [i for i in manifest["workloads"] if i["role"] == "test"]
    for item in tests:
        workload = Path(item["path"]).stem
        csv_path = run / f"test_model_{workload}.csv"
        row, records = read_predictions(csv_path)
        indices, labels, preds = map(np.asarray, zip(*records))
        assert indices.tolist() == list(range(item["selected"])), "Missing/duplicate test predictions"
        assert np.isfinite(labels).all() and np.isfinite(preds).all()
        assert (labels > 0).all() and (preds > 0).all(), "Runtime predictions must be positive"
        raw = json.loads((ROOT / item["path"]).read_text())
        np.testing.assert_allclose(labels, [p["plan_runtime"] / 1000 for p in raw["parsed_plans"]], rtol=1e-5)
        errors = np.maximum(labels / preds, preds / labels)
        values = np.percentile(errors, [50, 95, 100])
        for percentile, value in zip([50, 95, 100], values):
            np.testing.assert_allclose(float(row[f"val_median_q_error_{percentile}"]), value, rtol=1e-5)
        if reload_dir:
            _, reloaded = read_predictions(reload_dir / csv_path.name)
            np.testing.assert_allclose(np.array(records), np.array(reloaded), rtol=1e-5, atol=1e-6)
        results[workload] = {"queries": len(records), "median_q_error": float(values[0]),
                             "p95_q_error": float(values[1]), "max_q_error": float(values[2])}
        with (run / f"predictions_{workload}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_index", "runtime_seconds", "predicted_seconds", "q_error"])
            writer.writerows((*record, float(error)) for record, error in zip(records, errors))
    assert "imdb" not in manifest["train_databases"] and len(manifest["train_databases"]) == 19
    return results


def run_training(args):
    inputs, manifest = prepare(args.samples_per_workload, args.test_samples, args.seed)
    train_paths = [i["path"] for i in manifest["workloads"] if i["role"] == "train_validation"]
    test_paths = [i["path"] for i in manifest["workloads"] if i["role"] == "test"]
    run = ROOT / "output" / f"smoke_{args.mode}_s{args.seed}_n{args.samples_per_workload}_e{args.epochs}_t{args.test_samples}"
    run.mkdir(parents=True, exist_ok=True)
    config = ROOT / "setup" / "tuned_hyperparameters" / CONFIGS[args.mode]
    binding = {"mode": args.mode, "seed": args.seed, "epochs": args.epochs,
               "inputs_sha256": sha256(inputs / "manifest.json"), "config_sha256": sha256(config),
               "code": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT / "models").rglob("*.py"))},
               "train_py_sha256": sha256(ROOT / "train.py"), "threads": args.threads}
    binding_path = run / "binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("Code/config/input changed: refusing to reuse a mismatched checkpoint")
    write_json(binding_path, binding)
    command = [sys.executable, "train.py", "--train_model", "--workload_runs", *train_paths,
               "--test_workload_runs", *test_paths, "--statistics_file", str(inputs / "statistics.json"),
               "--target", str(run), "--filename_model", "model", "--hyperparameter_path", str(config),
               "--max_no_epochs", str(args.epochs), "--max_epoch_tuples", "100000", "--num_workers", "0",
               "--database", "postgres", "--device", "cpu", "--seed", str(args.seed)]
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               DGLBACKEND="pytorch", DGLDEFAULTDIR=str(ROOT / "output" / "dgl"),
               PYTHONHASHSEED=str(args.seed), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    started = time.perf_counter()
    write_json(run / "command.json", command)
    with (run / "train.log").open("a", encoding="utf-8") as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    verify(run, manifest)
    reload_dir = run / "reload"
    reload_dir.mkdir(exist_ok=True)
    for name in ["model.pt", "model.csv", "model_label_norm.pkl"]:
        shutil.copy2(run / name, reload_dir / name)
    # Independent process reload; remove only reproducible evaluation CSVs from prior checks.
    for old in reload_dir.glob("test_model_*.csv"):
        old.unlink()
    reload_command = command.copy()
    reload_command[reload_command.index("--target") + 1] = str(reload_dir)
    with (run / "reload.log").open("w", encoding="utf-8") as stream:
        subprocess.run(reload_command + ["--skip_train"], cwd=ROOT, env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    if "Successfully loaded checkpoint" not in (run / "reload.log").read_text(encoding="utf-8"):
        raise RuntimeError("Checkpoint was not reloaded")
    results = verify(run, manifest, reload_dir)
    # Reload copies are disposable; original model, logs, and predictions remain.
    assert reload_dir.resolve().is_relative_to((ROOT / "output").resolve())
    shutil.rmtree(reload_dir)
    with (run / "model.csv").open() as stream:
        epochs_completed = len(list(csv.DictReader(stream)))
    selected = sum(i["selected"] for i in manifest["workloads"] if i["role"] == "train_validation")
    summary = {"status": "PIPELINE_VERIFIED_NOT_FULL_PAPER_REPRODUCTION", "mode": args.mode,
               "seed": args.seed, "epochs_completed": epochs_completed, "device": "cpu",
               "train_databases": TRAIN_DBS, "held_out_database": "imdb", "selected_source_queries": selected,
               "training_queries": int(selected * 0.85), "validation_queries": selected - int(selected * 0.85),
               "statistics_scope": manifest["statistics_scope"], "test_results": results,
               "checkpoint_reload_matches": True, "metrics_independently_verified": True,
               "elapsed_seconds_this_invocation": time.perf_counter() - started,
               "checkpoint_sha256": sha256(run / "model.pt"), "run_directory": run.relative_to(ROOT).as_posix()}
    write_json(run / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=CONFIGS, default="postgres")
    parser.add_argument("--samples-per-workload", type=int, default=256, help="0 uses all available source queries")
    parser.add_argument("--test-samples", type=int, default=512, help="0 evaluates each complete test workload")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.samples_per_workload, args.test_samples) < 0 or args.epochs < 1 or args.threads < 1:
        parser.error("Sample caps must be nonnegative; epochs and threads must be positive")
    # Python's hash seed must be set before interpreter startup (categorical IDs use sets upstream).
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        environment = dict(os.environ, PYTHONHASHSEED=str(args.seed), PYTHONUTF8="1",
                           PYTHONDONTWRITEBYTECODE="1")
        sys.exit(subprocess.run([sys.executable, *sys.argv], env=environment).returncode)
    os.environ.setdefault("DGLBACKEND", "pytorch")
    os.environ.setdefault("DGLDEFAULTDIR", str(ROOT / "output" / "dgl"))
    run_training(args)
