"""Prevent accidental changes to the experiment implementation during relocation."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2] / "attacks/dropvla"
def test_pinned_sources_match_original_or_declared_path_adaptations():
    manifest = json.loads((ROOT / "reproduction_manifest.json").read_text())
    removed = manifest.get("removed_sources", {})
    for name, original in manifest["files_sha256"].items():
        if name in removed:
            assert removed[name]["sha256"] == original, name
            assert not (ROOT / name).exists(), name
            continue
        expected = manifest["relocation_changes"].get(name, {}).get("relocated", original)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert set(manifest["relocation_changes"]) == {"scripts/_common.sh", "scripts/wait_dropvla.sh", "scripts/validate_dropvla_cpu.py", "models/secure_vla.py", "utils/args.py", "scripts/download_assets.py"}
    # Security-tool removal must never rewrite the completed experiment's algorithms.
    for name in manifest["files_sha256"]:
        if name == "_main_original.py" or name.startswith("models/core/") or (
            name.startswith("utils/") and name != "utils/args.py"
        ) or name == "attacks/dropvla.py":
            assert name not in removed and name not in manifest["relocation_changes"], name

def test_assets_links_are_shared_and_relocatable():
    repo = ROOT.parents[1]
    for name in (".cache", "memoryvlasec", "output"):
        assert (ROOT / name).is_symlink()
        assert (ROOT / name).resolve() == (repo / name).resolve()
