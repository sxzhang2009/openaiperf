from __future__ import annotations

from pathlib import Path
from typing import Optional

import orjson
from sqlmodel import Session, select

from .models import Submission
from .store import Storage


def seed_if_empty(storage_dir: Path, session: Session) -> Optional[str]:
    count = session.exec(select(Submission)).first()
    if count:
        return None

    storage = Storage(storage_dir)
    # Build an example bundle from inline JSON (derived from OpenAIPerf.md example)
    uid_dir = storage_dir / "demo"
    uid_dir.mkdir(parents=True, exist_ok=True)

    system = {"cpu": "Xeon 8480+", "gpu": [{"name": "H100", "count": 8}], "driver": "550.54"}
    stack = {"framework": "pytorch==2.4.0", "cuda": "12.4", "engine": "tensorrt==10.0", "container": "sha256:demo"}
    model = {"name": "llama3-8b", "source": "hf://...", "dtype": "fp8", "quant": "awq", "weights_sha256": "demo", "license": "mit"}
    run = {
        "task": "llm.generation",
        "scenario": "server",
        "qps_target": 100,
        "context_len": 8192,
        "dataset": "evalhub://mixqna-v1",
        "subset_seed": 1234,
        "repeats": 3,
        "perf": {"throughput": 320.5, "latency_p99_ms": 95.1},
        "quality": {"achieved": 76.4},
    }
    energy = {"energy_per_request_j": 12.4, "avg_power_w": 510}

    (uid_dir / "system.json").write_bytes(orjson.dumps(system))
    (uid_dir / "stack.json").write_bytes(orjson.dumps(stack))
    (uid_dir / "model.json").write_bytes(orjson.dumps(model))
    (uid_dir / "run.json").write_bytes(orjson.dumps(run))
    (uid_dir / "energy_summary.json").write_bytes(orjson.dumps(energy))

    summary = storage.build_summary({"id": "demo", "dir": str(uid_dir)})

    sub = Submission(
        id="demo",
        task=summary.get("run", {}).get("task"),
        scenario=summary.get("run", {}).get("scenario"),
        model_name=summary.get("model", {}).get("name"),
        backend=summary.get("stack", {}).get("engine") or summary.get("stack", {}).get("framework"),
        quality=summary.get("metrics", {}).get("quality"),
        throughput=summary.get("metrics", {}).get("throughput"),
        latency_p99_ms=summary.get("metrics", {}).get("latency_p99_ms"),
        energy_j_per_request=summary.get("metrics", {}).get("energy_j_per_request"),
        cards=summary,
        notes="示例数据（自动生成）",
        approved=True,
    )
    session.add(sub)
    session.commit()
    return sub.id


