"""Download official preprocessed DBGen workloads, verify and extract locally."""
import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVES = {
    "deepdb_augmented": ("https://osf.io/download/vjpe5/", "5962490098f095113c655f4a6c200e11befdd92af58ad9e86797b2894359b8ff"),
    "parsed_plans": ("https://osf.io/download/2wtcg/", "edf38a68e1a1831f3fc995d24a8ff7f0cc56b25063e7575cb710d51c56782f9f"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(keep_archives=False):
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    runs = data / "runs"
    runs.mkdir(exist_ok=True)
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, (url, expected_hash) in ARCHIVES.items():
        previous = manifest.get(name, {})
        files = previous.get("files", {})
        if previous.get("archive_sha256") == expected_hash and files and all(
            (runs / p).is_file() and sha256(runs / p) == h for p, h in files.items()
        ):
            print(f"Verified existing {name}: {len(files)} files", flush=True)
            continue
        archive = data / f"{name}.zip"
        if not archive.exists():
            partial = archive.with_suffix(".zip.part")
            subprocess.run(["curl", "-L", "--fail", "--retry", "3", "--connect-timeout", "30",
                            "--max-time", "1800", "-C", "-", "-o", str(partial), url], check=True)
            partial.replace(archive)
        if sha256(archive) != expected_hash:
            raise ValueError(f"Checksum mismatch: {archive}; retained for inspection")
        files = {}
        with zipfile.ZipFile(archive) as zipped:
            bad = zipped.testzip()
            if bad:
                raise ValueError(f"ZIP CRC failure: {bad}")
            for member in zipped.infolist():
                if member.is_dir() or "__MACOSX" in member.filename or Path(member.filename).name == ".DS_Store":
                    continue
                target = (runs / member.filename).resolve()
                if not target.is_relative_to(runs.resolve()):
                    raise ValueError(f"Unsafe archive path: {member.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                files[target.relative_to(runs).as_posix()] = sha256(target)
        manifest[name] = {"url": url, "archive_sha256": expected_hash, "files": files}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if not keep_archives:
            archive.unlink()
        print(f"Verified and extracted {name}: {len(files)} files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-archives", action="store_true")
    download(parser.parse_args().keep_archives)
