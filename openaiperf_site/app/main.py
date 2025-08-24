from pathlib import Path
from typing import Optional

import orjson
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from starlette.middleware.sessions import SessionMiddleware

from .models import (
    init_db,
    get_session,
    Submission,
    Blog,
    User,
    pwd_context,
    Comment,
    engine,
    UserRole,
)
from sqlmodel import Session
from .store import Storage
from .seed import seed_if_empty
from .email_service import email_service


def get_current_user(request: Request) -> Optional[User]:
    username = request.session.get("user")
    if not username:
        return None
    with Session(engine) as session:
        user = session.query(User).filter(User.username == username).first()
        # Only return user if they are approved
        if user and user.approved:
            return user
        return None


def get_user_by_id(session, user_id: int) -> Optional[User]:
    """Helper function to get user by ID"""
    if not user_id:
        return None
    return session.get(User, user_id)


def get_admin_users(session) -> list[User]:
    """Get all admin users for notifications"""
    return session.query(User).filter(User.role == UserRole.ADMIN).all()


def send_new_submission_notification(session, submission: Submission, submitter: User, request: Request):
    """Send email notification to admins about new submission"""
    try:
        admin_users = get_admin_users(session)
        if not admin_users:
            print("No admin users found for notification")
            return
        
        # Build admin panel URL
        base_url = str(request.base_url).rstrip('/')
        admin_url = f"{base_url}/admin"
        
        for admin in admin_users:
            success = email_service.send_new_submission_notification(
                admin_email=admin.email,
                submission_id=submission.id,
                submitter_username=submitter.username,
                model_name=submission.model_name,
                admin_url=admin_url
            )
            
            if success:
                print(f"Notification sent to admin {admin.username} ({admin.email})")
            else:
                print(f"Failed to send notification to admin {admin.username} ({admin.email})")
                
    except Exception as e:
        print(f"Error sending submission notification: {str(e)}")


def send_registration_request_notification(session, user: User, request: Request):
    """Send email notification to admins about new registration request"""
    try:
        admin_users = get_admin_users(session)
        if not admin_users:
            print("No admin users found for registration notification")
            return
        
        # Build admin panel URL
        base_url = str(request.base_url).rstrip('/')
        admin_url = f"{base_url}/admin/users"
        
        for admin in admin_users:
            success = email_service.send_registration_request_notification(
                admin_email=admin.email,
                username=user.username,
                email=user.email,
                organization=user.organization,
                admin_url=admin_url
            )
            
            if success:
                print(f"Registration notification sent to admin {admin.username} ({admin.email})")
            else:
                print(f"Failed to send registration notification to admin {admin.username} ({admin.email})")
                
    except Exception as e:
        print(f"Error sending registration notification: {str(e)}")


def send_registration_approval_notification(user: User, request: Request):
    """Send email notification to user about registration approval"""
    try:
        base_url = str(request.base_url).rstrip('/')
        login_url = f"{base_url}/login"
        
        success = email_service.send_registration_approval_notification(
            user_email=user.email,
            username=user.username,
            login_url=login_url
        )
        
        if success:
            print(f"Approval notification sent to user {user.username} ({user.email})")
        else:
            print(f"Failed to send approval notification to user {user.username} ({user.email})")
            
    except Exception as e:
        print(f"Error sending approval notification: {str(e)}")


def get_user_for_template(request: Request) -> dict:
    """Dependency to pass user to templates."""
    return {"current_user": get_current_user(request)}


def require_user(request: Request) -> User:
    user = get_current_user(request)
    if not user:
        # Instead of raising an exception, redirect to the login page
        from urllib.parse import quote
        
        next_url = quote(str(request.url))
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            detail="Not authenticated",
            headers={"Location": f"/login?next={next_url}"},
        )
    return user


def require_auditor(user: User = Depends(require_user)) -> User:
    if user.role not in [UserRole.AUDITOR, UserRole.ADMIN]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied: Auditors only")
    return user


def require_admin(user: User = Depends(require_user)) -> User:
    if user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied: Admins only")
    return user


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
STORAGE_DIR = BASE_DIR / "storage"


app = FastAPI(title="OpenAIPerf Site")
app.add_middleware(SessionMiddleware, secret_key="your-secret-key-change-this-in-production")
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
def index(request: Request, user_ctx: dict = Depends(get_user_for_template)):
    ctx = {"request": request}
    ctx.update(user_ctx)
    return templates.TemplateResponse("index.html", ctx)


@app.get("/results", response_class=HTMLResponse)
def list_results(
    request: Request,
    session=Depends(get_session),
    q: Optional[str] = None,
    task: Optional[str] = None,
    scenario: Optional[str] = None,
    user_ctx: dict = Depends(get_user_for_template),
):
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
    ctx = {
        "request": request,
        "rows": rows,
        "q": q,
        "task": task,
        "scenario": scenario,
    }
    ctx.update(user_ctx)
    return templates.TemplateResponse("results.html", ctx)


def parse_system_log_for_charts(storage_dir: Path, submission_id: str):
    """Parse system.log to extract data for charts"""
    log_path = storage_dir / submission_id / "system.log"
    if not log_path.exists():
        return {
            "timestamps": [],
            "gpu_utilization": [],
            "power": [],
            "memory": [],
            "bandwidth": []
        }
    
    try:
        lines = log_path.read_text().strip().split('\n')
        timestamps = []
        gpu_utilization = []
        power = []
        memory = []
        bandwidth = []
        
        for line in lines:
            if not line.strip():
                continue
            try:
                data = orjson.loads(line)
                # Format timestamp for display
                timestamp = data.get('timestamp', '').split('T')[1].split('Z')[0]
                timestamps.append(timestamp)
                
                # GPU utilization (average across all GPUs)
                gpu_utils = data.get('gpu_utilization', [])
                avg_gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
                gpu_utilization.append(round(avg_gpu, 1))
                
                # Power consumption
                power.append(data.get('power_watts', 0))
                
                # Memory usage
                memory.append(data.get('memory_used_gb', 0))
                
                # Memory bandwidth (already in GB/s)
                bandwidth.append(round(data.get('memory_bandwidth_gbps', 0), 1))
                
            except:
                continue
                
        return {
            "timestamps": timestamps,
            "gpu_utilization": gpu_utilization,
            "power": power,
            "memory": memory,
            "bandwidth": bandwidth
        }
    except:
        return {
            "timestamps": [],
            "gpu_utilization": [],
            "power": [],
            "memory": [],
            "bandwidth": []
        }


@app.get("/results/{submission_id}", response_class=HTMLResponse)
def result_detail(
    submission_id: str,
    request: Request,
    session=Depends(get_session),
    user_ctx: dict = Depends(get_user_for_template),
):
    row = session.get(Submission, submission_id)
    if not row:
        return RedirectResponse(url="/results")
    
    # Add user information manually
    user_name = None
    if row.user_id:
        user = get_user_by_id(session, row.user_id)
        user_name = user.username if user else None
    
    # Parse system log data for charts
    system_log_data = parse_system_log_for_charts(STORAGE_DIR, submission_id)
    
    # Read run log content
    run_log_path = STORAGE_DIR / submission_id / "run.log"
    run_log_content = ""
    if run_log_path.exists():
        try:
            run_log_content = run_log_path.read_text()
        except:
            run_log_content = "无法读取运行日志文件"
    else:
        run_log_content = "运行日志文件不存在"
    
    ctx = {
        "request": request,
        "row": row,
        "user_name": user_name,
        "system_log_data": system_log_data,
        "run_log_content": run_log_content,
    }
    ctx.update(user_ctx)
    return templates.TemplateResponse("detail.html", ctx)


@app.get("/register", response_class=HTMLResponse)
def register_form(request: Request, user_ctx: dict = Depends(get_user_for_template)):
    ctx = {"request": request}
    ctx.update(user_ctx)
    return templates.TemplateResponse("register.html", ctx)


@app.post("/register")
def register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...),
    organization: Optional[str] = Form(default=None),
    phone: Optional[str] = Form(default=None),
    session=Depends(get_session),
):
    # Check if user or email already exists
    existing_user = session.query(User).filter((User.username == username) | (User.email == email)).first()
    if existing_user:
        error_msg = "Username already exists." if existing_user.username == username else "Email already registered."
        ctx = {"request": request, "error": error_msg}
        return templates.TemplateResponse("register.html", ctx, status_code=409)

    hashed_password = pwd_context.hash(password)
    user = User(
        username=username,
        hashed_password=hashed_password,
        email=email,
        organization=organization,
        phone=phone,
        approved=False,  # Create user in pending state
    )
    session.add(user)
    session.commit()
    
    # Send registration request notification to admins
    send_registration_request_notification(session, user, request)
    
    # Show pending approval message instead of logging in
    ctx = {"request": request, "success": "Registration request submitted successfully! Please wait for admin approval. You will receive an email notification when your account is approved."}
    return templates.TemplateResponse("register.html", ctx)


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request, user_ctx: dict = Depends(get_user_for_template)):
    ctx = {"request": request, "error": None}
    ctx.update(user_ctx)
    return templates.TemplateResponse("login.html", ctx)


@app.post("/login")
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),  # Capture the 'next' URL from the form
    session=Depends(get_session),
):
    user = session.query(User).filter(User.username == username).first()
    if not user or not user.verify_password(password):
        return templates.TemplateResponse(
            "login.html", {"request": request, "error": "Invalid username or password"}
        )
    request.session["user"] = user.username

    # Redirect to the 'next' URL if provided, otherwise default based on role
    if next and next != "/":
        return RedirectResponse(url=next, status_code=status.HTTP_303_SEE_OTHER)
    
    if user.role in [UserRole.ADMIN, UserRole.AUDITOR]:
        return RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/logout")
def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/blog", response_class=HTMLResponse)
def blog_list(request: Request, session=Depends(get_session), user_ctx: dict = Depends(get_user_for_template)):
    rows = session.query(Blog).order_by(Blog.created_at.desc()).all()
    ctx = {"request": request, "rows": rows}
    ctx.update(user_ctx)
    return templates.TemplateResponse("blog/list.html", ctx)


@app.get("/blog/{blog_id}", response_class=HTMLResponse)
def blog_detail(
    blog_id: int,
    request: Request,
    session=Depends(get_session),
    user_ctx: dict = Depends(get_user_for_template),
):
    row = session.get(Blog, blog_id)
    if not row:
        return RedirectResponse(url="/blog")
    
    # Get comments with usernames manually
    comments = session.query(Comment, User.username).join(User, Comment.user_id == User.id).filter(Comment.blog_id == blog_id).all()
    comment_data = [{"content": c.content, "created_at": c.created_at, "username": username} for c, username in comments]
    
    ctx = {"request": request, "row": row, "comments": comment_data}
    ctx.update(user_ctx)
    return templates.TemplateResponse("blog/detail.html", ctx)


@app.post("/blog/{blog_id}/comment")
def add_comment(
    blog_id: int,
    content: str = Form(...),
    user: User = Depends(require_user),
    session=Depends(get_session),
):
    comment = Comment(content=content, user_id=user.id, blog_id=blog_id)
    session.add(comment)
    session.commit()
    return RedirectResponse(url=f"/blog/{blog_id}", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/archive", response_class=HTMLResponse)
def archive(request: Request, user_ctx: dict = Depends(get_user_for_template)):
    ctx = {"request": request}
    ctx.update(user_ctx)
    return templates.TemplateResponse("blog/archive.html", ctx)


@app.get("/submit", response_class=HTMLResponse)
def submit_form(request: Request, user: User = Depends(require_user), user_ctx: dict = Depends(get_user_for_template)):
    ctx = {"request": request}
    ctx.update(user_ctx)
    return templates.TemplateResponse("submit.html", ctx)


def generate_submission_id(session) -> str:
    """Generate submission ID in format: 25H1-001"""
    import datetime
    
    now = datetime.datetime.now()
    year = str(now.year)[-2:]  # Last 2 digits of year
    half = f"H{1 if now.month <= 6 else 2}"  # H1 for Jan-Jun, H2 for Jul-Dec
    
    # Find the highest sequence number for this year-half
    prefix = f"{year}{half}-"
    existing = session.query(Submission).filter(Submission.id.like(f"{prefix}%")).all()
    
    if not existing:
        seq_num = 1
    else:
        # Extract sequence numbers and find the max
        seq_nums = []
        for sub in existing:
            try:
                seq_part = sub.id.split('-')[1]
                seq_nums.append(int(seq_part))
            except (IndexError, ValueError):
                continue
        seq_num = max(seq_nums) + 1 if seq_nums else 1
    
    return f"{prefix}{seq_num:03d}"


@app.post("/submit")
async def submit(
    request: Request,
    system: Optional[UploadFile] = File(default=None),
    stack: Optional[UploadFile] = File(default=None),
    model: Optional[UploadFile] = File(default=None),
    run: Optional[UploadFile] = File(default=None),
    system_log: Optional[UploadFile] = File(default=None),
    run_log: Optional[UploadFile] = File(default=None),
    notes: Optional[str] = Form(default=""),
    session=Depends(get_session),
    user: User = Depends(require_user),
):
    # Generate custom submission ID
    submission_id = generate_submission_id(session)
    
    storage = Storage(STORAGE_DIR)
    saved = await storage.save_bundle(system=system, stack=stack, model=model, run=run, system_log=system_log, run_log=run_log)

    summary = storage.build_summary(saved)
    submission = Submission(
        id=submission_id,  # Use our custom ID instead of saved["id"]
        user_id=user.id,  # Associate submission with the current user
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

    # Send email notification to admins
    send_new_submission_notification(session, submission, user, request)

    return RedirectResponse(url=f"/results/{submission.id}", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/profile", response_class=HTMLResponse)
def user_profile(
    request: Request,
    user: User = Depends(require_user),
    session=Depends(get_session),
    user_ctx: dict = Depends(get_user_for_template),
):
    # Query submissions by the current user
    submissions = session.query(Submission).filter(Submission.user_id == user.id).order_by(Submission.created_at.desc()).all()
    
    ctx = {
        "request": request,
        "user": user,
        "submissions": submissions,
    }
    ctx.update(user_ctx)
    return templates.TemplateResponse("profile.html", ctx)


@app.post("/submission/{submission_id}/delete")
def delete_submission(
    submission_id: str,
    request: Request,
    user: User = Depends(require_user),
    session=Depends(get_session),
):
    submission = session.get(Submission, submission_id)
    if not submission:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Submission not found")

    # Security check: User can only delete their own unapproved submissions
    if submission.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You can only delete your own submissions")
    if submission.approved:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete an approved submission")

    # Delete associated files from storage first
    try:
        storage = Storage(STORAGE_DIR)
        storage.delete_bundle(submission_id)
    except Exception as e:
        # Log the error but proceed to delete the DB record anyway
        print(f"Error deleting files for submission {submission_id}: {e}")

    session.delete(submission)
    session.commit()

    return RedirectResponse(url="/profile", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard(
    request: Request,
    session=Depends(get_session),
    user: User = Depends(require_auditor),
    user_ctx: dict = Depends(get_user_for_template),
):
    total = session.query(Submission).count()
    latest_submissions = session.query(Submission).order_by(Submission.created_at.desc()).limit(10).all()
    pending_submissions = session.query(Submission).filter(Submission.approved == False).order_by(Submission.created_at.desc()).all()  # noqa: E712
    
    # Add user information manually by creating enhanced objects
    def enhance_with_user_info(submissions):
        enhanced = []
        for submission in submissions:
            # Create a dict with submission data and user info
            enhanced_submission = {
                'id': submission.id,
                'created_at': submission.created_at,
                'task': submission.task,
                'model_name': submission.model_name,
                'backend': submission.backend,
                'approved': submission.approved,
                'user_id': submission.user_id,
                'user_name': None
            }
            
            if submission.user_id:
                user = get_user_by_id(session, submission.user_id)
                enhanced_submission['user_name'] = user.username if user else None
            
            enhanced.append(enhanced_submission)
        return enhanced
    
    latest = enhance_with_user_info(latest_submissions)
    pending = enhance_with_user_info(pending_submissions)
    
    ctx = {
        "request": request,
        "total": total,
        "latest": latest,
        "pending": pending,
    }
    ctx.update(user_ctx)
    return templates.TemplateResponse("admin.html", ctx)


@app.get("/admin/blog", response_class=HTMLResponse)
def admin_blog(
    request: Request,
    session=Depends(get_session),
    user: User = Depends(require_admin),
    user_ctx: dict = Depends(get_user_for_template),
):
    rows = session.query(Blog).order_by(Blog.created_at.desc()).all()
    ctx = {"request": request, "rows": rows}
    ctx.update(user_ctx)
    return templates.TemplateResponse("admin_blog.html", ctx)


@app.get("/user/{user_id}", response_class=HTMLResponse)
def user_info(
    user_id: int,
    request: Request,
    session=Depends(get_session),
    user: User = Depends(require_auditor),  # Only auditors and admins can view user info
    user_ctx: dict = Depends(get_user_for_template),
):
    """Display user information and their submissions"""
    target_user = get_user_by_id(session, user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get user's submissions
    submissions = session.query(Submission).filter(Submission.user_id == user_id).order_by(Submission.created_at.desc()).all()
    
    ctx = {
        "request": request,
        "target_user": target_user,
        "submissions": submissions,
        "submission_count": len(submissions),
        "approved_count": len([s for s in submissions if s.approved]),
    }
    ctx.update(user_ctx)
    return templates.TemplateResponse("user_info.html", ctx)


@app.get("/admin/users", response_class=HTMLResponse)
def admin_users_list(
    request: Request,
    session=Depends(get_session),
    user: User = Depends(require_admin),
    user_ctx: dict = Depends(get_user_for_template),
):
    users = session.query(User).order_by(User.id).all()
    ctx = {"request": request, "users": users}
    ctx.update(user_ctx)
    return templates.TemplateResponse("admin/users.html", ctx)


@app.post("/admin/user/{user_id}/approve")
def admin_approve_user(
    user_id: int,
    request: Request,
    session=Depends(get_session),
    current_user: User = Depends(require_admin),
):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.approved:
        raise HTTPException(status_code=400, detail="User already approved")
    
    # Approve the user
    user.approved = True
    session.commit()
    
    # Send approval notification to user
    send_registration_approval_notification(user, request)
    
    return RedirectResponse(url="/admin/users", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/admin/user/{user_id}/delete")
def admin_delete_user(
    user_id: int,
    request: Request,
    session=Depends(get_session),
    current_user: User = Depends(require_admin),
):
    if current_user.id == user_id:
        raise HTTPException(status_code=400, detail="Admins cannot delete themselves.")

    user_to_delete = session.get(User, user_id)
    if not user_to_delete:
        raise HTTPException(status_code=404, detail="User not found.")
    
    if user_to_delete.role == UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Cannot delete another admin account.")

    # Anonymize their submissions instead of deleting them
    submissions = session.query(Submission).filter(Submission.user_id == user_id).all()
    for sub in submissions:
        sub.user_id = None
        session.add(sub)

    session.delete(user_to_delete)
    session.commit()

    return RedirectResponse(url="/admin/users", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/admin/blog")
def admin_add_blog(
    request: Request,
    title: str = Form(...),
    content: str = Form(...),
    session=Depends(get_session),
    user: User = Depends(require_admin),
):
    blog = Blog(title=title, content=content)
    session.add(blog)
    session.commit()
    return RedirectResponse(url="/admin/blog", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/admin/revoke/{submission_id}")
def admin_revoke(
    submission_id: str,
    request: Request,
    session=Depends(get_session),
    user: User = Depends(require_auditor),
):
    row = session.get(Submission, submission_id)
    if not row:
        raise HTTPException(status_code=404)
    row.approved = False
    session.add(row)
    session.commit()
    return RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/admin/approve/{submission_id}")
def admin_approve(
    submission_id: str,
    request: Request,
    session=Depends(get_session),
    user: User = Depends(require_auditor),
):
    row = session.get(Submission, submission_id)
    if not row:
        raise HTTPException(status_code=404)
    row.approved = True
    session.add(row)
    session.commit()
    return RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)


