"""Prevent accidental changes to the experiment implementation during relocation."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2] / "attacks/dropvla"
def test_pinned_sources_match_original_or_declared_path_adaptations():
    manifest = json.loads((ROOT / "reproduction_manifest.json").read_text())
    for name, original in manifest["files_sha256"].items():
        expected = manifest["relocation_changes"].get(name, {}).get("relocated", original)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert set(manifest["relocation_changes"]) == {"scripts/_common.sh", "scripts/wait_dropvla.sh", "scripts/validate_dropvla_cpu.py"}

def test_assets_links_are_shared_and_relocatable():
    repo = ROOT.parents[1]
    for name in (".cache", "memoryvlasec", "output"):
        assert (ROOT / name).is_symlink()
        assert (ROOT / name).resolve() == (repo / name).resolve()
