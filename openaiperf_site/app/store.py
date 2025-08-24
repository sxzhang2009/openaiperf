from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path
from typing import Dict, Optional

import orjson
from fastapi import UploadFile


class Storage:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    async def save_bundle(
        self,
        *,
        system: Optional[UploadFile],
        stack: Optional[UploadFile],
        model: Optional[UploadFile],
        run: Optional[UploadFile],
        energy: Optional[UploadFile] = None,
        events: Optional[UploadFile] = None,
    ) -> Dict[str, str]:
        uid = uuid.uuid4().hex[:12]
        dir_path = self.base_dir / uid
        dir_path.mkdir(parents=True, exist_ok=True)

        saved: Dict[str, str] = {"id": uid, "dir": str(dir_path)}
        for name, obj in {
            "system.json": system,
            "stack.json": stack,
            "model.json": model,
            "run.json": run,
            "energy_summary.json": energy,
            "events.jsonl": events,
        }.items():
            if obj is None:
                continue
            content = await obj.read()
            (dir_path / name).write_bytes(content)
            saved[name] = str(dir_path / name)
        return saved

    def _load_json(self, path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            return orjson.loads(path.read_bytes())
        except Exception:
            return None

    def build_summary(self, saved: Dict[str, str]) -> dict:
        dir_path = Path(saved["dir"])  # type: ignore
        system = self._load_json(dir_path / "system.json") or {}
        stack = self._load_json(dir_path / "stack.json") or {}
        model = self._load_json(dir_path / "model.json") or {}
        run = self._load_json(dir_path / "run.json") or {}
        energy = self._load_json(dir_path / "energy_summary.json") or {}

        metrics: Dict[str, float] = {}
        # Heuristic extraction for MVP; can be replaced with schema-aware parsing later
        if energy:
            for key in ("energy_per_request_j", "energy_per_sample_j", "energy_per_token_j"):
                val = energy.get(key)
                if isinstance(val, (int, float)):
                    metrics["energy_j_per_request"] = float(val)
                    break
            if isinstance(energy.get("avg_power_w"), (int, float)):
                metrics["avg_power_w"] = float(energy["avg_power_w"])  # noqa: F841 (may be unused)

        perf = run.get("perf", {}) if isinstance(run.get("perf"), dict) else {}
        if isinstance(perf.get("throughput"), (int, float)):
            metrics["throughput"] = float(perf["throughput"])
        if isinstance(perf.get("latency_p99_ms"), (int, float)):
            metrics["latency_p99_ms"] = float(perf["latency_p99_ms"])

        quality = run.get("quality", {}) if isinstance(run.get("quality"), dict) else {}
        # Allow a direct numeric like {"metric": 76.4} or target/achieved
        cand = quality.get("achieved") or quality.get("metric")
        if isinstance(cand, (int, float)):
            metrics["quality"] = float(cand)

        summary = {
            "system": system,
            "stack": stack,
            "model": model,
            "run": run,
            "energy": energy,
            "metrics": metrics,
        }
        (dir_path / "summary.json").write_bytes(orjson.dumps(summary))
        return summary


