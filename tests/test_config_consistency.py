from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_theory_of_mind_config_names_are_consistent():
    for path in CONFIG_DIR.glob("*.yaml"):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        toggles = cfg.get("toggles") or {}

        assert "theory_of_mind" not in toggles

        tom_cfg = toggles.get("tom")
        tom_enabled = bool(tom_cfg and tom_cfg.get("enabled"))
        should_enable_tom = "tom" in path.stem or path.stem == "run_full_stack"

        assert tom_enabled == should_enable_tom
