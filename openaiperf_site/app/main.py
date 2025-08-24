from pathlib import Path
from typing import Optional

import orjson
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import status

from .models import init_db, get_session, Submission
from .store import Storage
from .seed import seed_if_empty


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
STORAGE_DIR = BASE_DIR / "storage"


app = FastAPI(title="OpenAIPerf Site")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.on_event("startup")
def on_startup() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    # Seed demo data once
    from sqlmodel import Session as SQLSession
    from .models import engine
    with SQLSession(engine) as s:
        seed_if_empty(STORAGE_DIR, s)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/results", response_class=HTMLResponse)
def list_results(request: Request, session=Depends(get_session), q: Optional[str] = None, task: Optional[str] = None, scenario: Optional[str] = None):
    query = session.query(Submission)
    query = query.filter(Submission.approved == True)  # noqa: E712
    if q:
        like = f"%{q}%"
        query = query.filter(Submission.model_name.ilike(like))
    if task:
        query = query.filter(Submission.task == task)
    if scenario:
        query = query.filter(Submission.scenario == scenario)
    rows = query.order_by(Submission.created_at.desc()).limit(200).all()
    return templates.TemplateResponse("results.html", {"request": request, "rows": rows, "q": q, "task": task, "scenario": scenario})


@app.get("/results/{submission_id}", response_class=HTMLResponse)
def result_detail(submission_id: str, request: Request, session=Depends(get_session)):
    row = session.get(Submission, submission_id)
    if not row:
        return RedirectResponse(url="/results")
    return templates.TemplateResponse("detail.html", {"request": request, "row": row})


@app.get("/submit", response_class=HTMLResponse)
def submit_form(request: Request):
    return templates.TemplateResponse("submit.html", {"request": request})


@app.post("/submit")
async def submit(
    request: Request,
    system: Optional[UploadFile] = File(default=None),
    stack: Optional[UploadFile] = File(default=None),
    model: Optional[UploadFile] = File(default=None),
    run: Optional[UploadFile] = File(default=None),
    energy: Optional[UploadFile] = File(default=None),
    events: Optional[UploadFile] = File(default=None),
    notes: Optional[str] = Form(default=""),
    session=Depends(get_session),
):
    storage = Storage(STORAGE_DIR)
    saved = await storage.save_bundle(system=system, stack=stack, model=model, run=run, energy=energy, events=events)

    summary = storage.build_summary(saved)
    submission = Submission(
        id=saved["id"],
        task=summary.get("run", {}).get("task"),
        scenario=summary.get("run", {}).get("scenario"),
        model_name=summary.get("model", {}).get("name"),
        backend=summary.get("stack", {}).get("engine") or summary.get("stack", {}).get("framework"),
        quality=summary.get("metrics", {}).get("quality"),
        throughput=summary.get("metrics", {}).get("throughput"),
        latency_p99_ms=summary.get("metrics", {}).get("latency_p99_ms"),
        energy_j_per_request=summary.get("metrics", {}).get("energy_j_per_request"),
        cards=summary,
        notes=notes or summary.get("run", {}).get("notes"),
        approved=False,
    )
    session.add(submission)
    session.commit()

    return RedirectResponse(url=f"/results/{submission.id}", status_code=303)


def is_admin(request: Request) -> bool:
    return request.cookies.get("oa_admin") == "1"


@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_form(request: Request):
    return templates.TemplateResponse("admin_login.html", {"request": request, "error": None})


@app.post("/admin/login")
def admin_login(request: Request, username: str = Form(...), password: str = Form(...)):
    # Simple credentials; replace with env vars or config in production
    if username == "admin" and password == "admin123":
        resp = RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)
        resp.set_cookie("oa_admin", "1", httponly=True, samesite="lax")
        return resp
    return templates.TemplateResponse("admin_login.html", {"request": request, "error": "用户名或密码错误"})


@app.get("/admin/logout")
def admin_logout():
    resp = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    resp.delete_cookie("oa_admin")
    return resp


@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard(request: Request, session=Depends(get_session)):
    if not is_admin(request):
        return RedirectResponse(url="/admin/login")
    total = session.query(Submission).count()
    latest = session.query(Submission).order_by(Submission.created_at.desc()).limit(10).all()
    pending = session.query(Submission).filter(Submission.approved == False).order_by(Submission.created_at.desc()).all()  # noqa: E712
    return templates.TemplateResponse("admin.html", {"request": request, "total": total, "latest": latest, "pending": pending})


@app.post("/admin/approve/{submission_id}")
def admin_approve(submission_id: str, request: Request, session=Depends(get_session)):
    if not is_admin(request):
        raise HTTPException(status_code=403)
    row = session.get(Submission, submission_id)
    if not row:
        raise HTTPException(status_code=404)
    row.approved = True
    session.add(row)
    session.commit()
    return RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)


