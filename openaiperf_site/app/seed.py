from __future__ import annotations

from pathlib import Path
from typing import Optional

import orjson
from sqlmodel import Session, select

from .models import Submission, Blog, User, UserRole, pwd_context
from .store import Storage
import random


def create_sample_submission(storage_dir: Path, session: Session, sub_id: str, org: str, system_name: str, 
                           cpu: str, cpu_count: int, accelerator_name: str, accelerator_count: int,
                           model_name: str, task: str, scenario: str, throughput: float, 
                           latency: float, quality: float, framework: str, user_id: int, accelerator_vendor: str = "nvidia") -> None:
    """Create a sample submission with given parameters"""
    uid_dir = storage_dir / sub_id
    uid_dir.mkdir(parents=True, exist_ok=True)

    # Determine accelerator type based on vendor
    if accelerator_vendor == "google":
        accelerator_type = "tpu"
        driver = "tpu_driver"
    else:
        accelerator_type = "gpu"
        driver = "550.54"
    
    system = {
        "organization": org,
        "system_name": system_name,
        "hardware": {
            "cpu": {"model": cpu, "cores": cpu_count},
            "accelerators": [{"type": accelerator_type, "vendor": accelerator_vendor, "name": accelerator_name, "count": accelerator_count}]
        },
        "software": {"driver": driver}
    }
    # Determine compute backend based on framework
    if "jax" in framework.lower():
        compute_backend = {"type": "xla", "version": "0.4.0"}
        engine = "jax==0.4.23"
    else:
        compute_backend = {"type": "cuda", "version": "12.4"}
        engine = "tensorrt==10.0"
    
    stack = {
        "framework": framework, 
        "compute_backend": compute_backend, 
        "engine": engine, 
        "container": f"sha256:{sub_id}"
    }
    model = {"name": model_name, "source": "hf://...", "dtype": "fp8", "quant": "awq", "weights_sha256": sub_id, "license": "mit"}
    run = {
        "task": task,
        "scenario": scenario,
        "qps_target": 100,
        "context_len": 8192,
        "dataset": "evalhub://mixqna-v1",
        "subset_seed": 1234,
        "repeats": 3,
        "perf": {"throughput": throughput, "latency_p99_ms": latency},
        "quality": {"achieved": quality},
    }

    (uid_dir / "system.json").write_bytes(orjson.dumps(system))
    (uid_dir / "stack.json").write_bytes(orjson.dumps(stack))
    (uid_dir / "model.json").write_bytes(orjson.dumps(model))
    (uid_dir / "run.json").write_bytes(orjson.dumps(run))

    # Create sample system.log
    system_log_content = ""
    for i in range(8):
        log_entry = {
            "timestamp": f"2025-01-04T10:29:{45+i:02d}Z",
            "cpu_usage": 82.1 + i * 0.5,
            "memory_used_gb": 42.3 + i * 0.2,
            "accelerator_utilization": [89.2 + i * 0.3] * accelerator_count,
            "accelerator_memory_used_gb": [76.2 + i * 0.1] * accelerator_count,
            "memory_bandwidth_gbps": 1024.5 + i * 2.1,
            "power_watts": 2840 + i * 5
        }
        system_log_content += orjson.dumps(log_entry).decode() + "\n"
    (uid_dir / "system.log").write_text(system_log_content)

    # Create sample run.log
    run_log_content = f"""2025-01-04T10:29:40Z [INFO] Starting OpenAIPerf evaluation...
2025-01-04T10:29:40Z [INFO] Configuration loaded: {task} task, {scenario} scenario
2025-01-04T10:29:41Z [INFO] Initializing {framework.split('==')[0]} engine...
2025-01-04T10:29:43Z [INFO] Loading model: {model_name}
2025-01-04T10:29:45Z [INFO] Model loaded successfully, dtype: fp8
2025-01-04T10:29:45Z [INFO] Quantization: AWQ enabled
2025-01-04T10:29:46Z [INFO] Starting inference server
2025-01-04T10:29:47Z [INFO] Beginning evaluation run (3 repeats)
2025-01-04T10:29:50Z [METRIC] Repeat 1/3 - Throughput: {throughput:.1f} req/s
2025-01-04T10:29:53Z [METRIC] Repeat 2/3 - Throughput: {throughput+1:.1f} req/s
2025-01-04T10:29:56Z [METRIC] Repeat 3/3 - Throughput: {throughput-0.5:.1f} req/s
2025-01-04T10:29:56Z [RESULT] Final metrics - Throughput: {throughput} req/s, P99 Latency: {latency}ms
2025-01-04T10:29:56Z [RESULT] Quality score: {quality}/100
2025-01-04T10:29:57Z [INFO] Evaluation completed successfully"""
    (uid_dir / "run.log").write_text(run_log_content)

    storage = Storage(storage_dir)
    summary = storage.build_summary({"id": sub_id, "dir": str(uid_dir)})

    sub = Submission(
        id=sub_id,
        user_id=user_id,
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


def seed_users(session: Session) -> None:
    """Create initial users if none exist"""
    if session.exec(select(User)).first():
        return

    users = [
        User(
            username="admin",
            hashed_password=pwd_context.hash("admin123"),
            role=UserRole.ADMIN,
            email="admin@openaiperf.org",
            organization="Admin Team",
            approved=True,
        ),
        User(
            username="auditor",
            hashed_password=pwd_context.hash("auditor123"),
            role=UserRole.AUDITOR,
            email="auditor@openaiperf.com",
            organization="Auditor Team",
            approved=True,
        ),
        User(
            username="user",
            hashed_password=pwd_context.hash("user123"),
            role=UserRole.USER,
            email="user@openaiperf.com",
            organization="Community",
            approved=True,
        ),
    ]
    for user in users:
        session.add(user)


def update_submission_ids_to_half_yearly(session: Session) -> None:
    """Update submission IDs from quarterly (25Q1) to half-yearly (25H1) format"""
    # Get all submissions with quarterly format
    quarterly_submissions = session.exec(
        select(Submission).where(Submission.id.like("%Q%"))
    ).all()
    
    if not quarterly_submissions:
        print("No submissions found with quarterly ID format")
        return
    
    print(f"Found {len(quarterly_submissions)} submissions with quarterly ID format")
    
    # Map quarters to halves
    quarter_to_half = {
        "Q1": "H1", "Q2": "H1",  # Q1, Q2 -> H1 (first half)
        "Q3": "H2", "Q4": "H2"   # Q3, Q4 -> H2 (second half)
    }
    
    updated_submissions = []
    for submission in quarterly_submissions:
        old_id = submission.id
        # Parse the ID format: 25Q1-001
        parts = old_id.split('-')
        if len(parts) == 2:
            year_quarter = parts[0]
            seq = parts[1]
            
            # Extract quarter
            for quarter, half in quarter_to_half.items():
                if quarter in year_quarter:
                    new_year_half = year_quarter.replace(quarter, half)
                    new_id = f"{new_year_half}-{seq}"
                    
                    # Check if new ID already exists
                    existing = session.get(Submission, new_id)
                    if not existing:
                        submission.id = new_id
                        updated_submissions.append((old_id, new_id))
                        print(f"Updated ID: {old_id} -> {new_id}")
                    else:
                        print(f"ID {new_id} already exists, skipping {old_id}")
                    break
    
    session.commit()
    print(f"Updated {len(updated_submissions)} submission IDs to half-yearly format")


def update_submissions_with_users(session: Session) -> None:
    """Update existing submissions that have null user_id with random users"""
    # Get all submissions with null user_id
    submissions_without_users = session.exec(
        select(Submission).where(Submission.user_id.is_(None))
    ).all()
    
    if not submissions_without_users:
        print("No submissions found with empty user_id")
        return
    
    # Get all existing users
    users = session.exec(select(User)).all()
    if not users:
        print("No users found in database")
        return
    
    print(f"Found {len(submissions_without_users)} submissions without user_id")
    print(f"Available users: {[u.username for u in users]}")
    
    # Assign random users to submissions
    for submission in submissions_without_users:
        # Assign a random user
        assigned_user = random.choice(users)
        submission.user_id = assigned_user.id
        print(f"Assigned submission {submission.id} to user {assigned_user.username}")
    
    session.commit()
    print(f"Updated {len(submissions_without_users)} submissions with user assignments")


def seed_if_empty(storage_dir: Path, session: Session) -> Optional[str]:
    # Seed users first
    seed_users(session)

    if session.exec(select(Submission)).first():
        return None

    # Get users to assign to submissions
    admin_user = session.exec(select(User).where(User.username == "admin")).first()
    auditor_user = session.exec(select(User).where(User.username == "auditor")).first()
    regular_user = session.exec(select(User).where(User.username == "user")).first()
    
    # Create multiple sample submissions with user assignments
    submissions = [
        ("25H1-001", "OpenAI Research", "DGX-H100-8x", "Intel Xeon 8480+", 56, "H100", 8, 
         "llama3-8b", "llm.generation", "server", 320.5, 95.1, 76.4, "pytorch==2.4.0", admin_user.id, "nvidia"),
        ("25H1-002", "Google DeepMind", "TPU-v5-256", "AMD EPYC 9B14", 128, "TPU-v5", 256, 
         "gemma-7b", "llm.generation", "offline", 1250.2, 45.3, 78.9, "jax==0.4.23", auditor_user.id, "google"),
        ("25H1-003", "NVIDIA Research", "DGX-A100-4x", "Intel Xeon 8462Y+", 48, "A100", 4, 
         "llama3-13b", "llm.generation", "server", 185.7, 124.6, 82.1, "pytorch==2.3.1", regular_user.id, "nvidia"),
        ("25H1-004", "Microsoft Azure", "ND96amsr_A100_v4", "AMD EPYC 7V12", 96, "A100", 8, 
         "phi-3-mini", "llm.generation", "offline", 890.3, 28.7, 74.2, "onnxruntime==1.17.0", admin_user.id, "nvidia"),
        ("25H1-005", "Meta AI", "RSC-A100-16x", "Intel Xeon Platinum 8380", 80, "A100", 16, 
         "llama2-70b", "llm.generation", "server", 95.8, 256.3, 85.6, "pytorch==2.2.0", auditor_user.id, "nvidia"),
    ]

    for sub_data in submissions:
        create_sample_submission(storage_dir, session, *sub_data)

    # Create sample blog posts
    blog1 = Blog(title="OpenAIPerf 新版本发布：支持四卡二日志", 
                content="我们很高兴宣布 OpenAIPerf 新版本正式发布！本次更新引入了全新的'四卡二日志'提交格式，提供更详细的系统监控和运行日志分析功能。")
    blog2 = Blog(title="AI 系统性能评测最佳实践", 
                content="在进行 AI 系统性能评测时，需要考虑多个维度的指标。本文将介绍如何设计全面的评测方案，包括吞吐量、延迟、质量和能耗等关键指标。")
    blog3 = Blog(title="深度解析：GPU 利用率与内存带宽优化", 
                content="GPU 利用率和内存带宽是影响 AI 模型推理性能的关键因素。通过系统日志分析，我们可以识别性能瓶颈并进行针对性优化。")
    
    session.add(blog1)
    session.add(blog2)
    session.add(blog3)

    session.commit()
    return "25Q1-001"


