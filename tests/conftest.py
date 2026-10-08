import os
from pathlib import Path

os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGLDEFAULTDIR", str(Path(__file__).resolve().parents[1] / "output" / "dgl"))
