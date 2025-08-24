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
        system_log: Optional[UploadFile] = None,
        run_log: Optional[UploadFile] = None,
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
            "system.log": system_log,
            "run.log": run_log,
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

    def _parse_system_log(self, path: Path) -> dict:
        """Parse system.log to extract power and performance metrics"""
        if not path.exists():
            return {}
        
        try:
            lines = path.read_text().strip().split('\n')
            total_power = 0
            count = 0
            max_gpu_util = 0
            max_memory_bw = 0
            
            for line in lines:
                if not line.strip():
                    continue
                try:
                    data = orjson.loads(line)
                    if 'power_watts' in data:
                        total_power += data['power_watts']
                        count += 1
                    if 'gpu_utilization' in data:
                        max_gpu_util = max(max_gpu_util, max(data['gpu_utilization']))
                    if 'memory_bandwidth_gbps' in data:
                        max_memory_bw = max(max_memory_bw, data['memory_bandwidth_gbps'])
                except:
                    continue
            
            metrics = {}
            if count > 0:
                metrics['avg_power_w'] = total_power / count
            if max_gpu_util > 0:
                metrics['max_gpu_utilization'] = max_gpu_util
            if max_memory_bw > 0:
                metrics['max_memory_bandwidth_gbps'] = max_memory_bw
                
            return metrics
        except:
            return {}

    def build_summary(self, saved: Dict[str, str]) -> dict:
        dir_path = Path(saved["dir"])  # type: ignore
        system = self._load_json(dir_path / "system.json") or {}
        stack = self._load_json(dir_path / "stack.json") or {}
        model = self._load_json(dir_path / "model.json") or {}
        run = self._load_json(dir_path / "run.json") or {}
        
        # Parse system.log for power and performance metrics
        system_log_metrics = self._parse_system_log(dir_path / "system.log")

        metrics: Dict[str, float] = {}
        
        # Extract metrics from system.log
        metrics.update(system_log_metrics)

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
            "system_log_metrics": system_log_metrics,
            "metrics": metrics,
        }
        (dir_path / "summary.json").write_bytes(orjson.dumps(summary))
        return summary


