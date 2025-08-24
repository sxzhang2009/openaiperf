OpenAIPerf Site (MVP)

Stack
- Backend: FastAPI
- Templates: Jinja2 + Tailwind (CDN) for a clean, modern look
- DB: SQLite via SQLModel

Features
- Landing page aligned with the OpenAIPerf vision
- Results matrix list with simple filters (task, scenario, model, backend)
- Result detail page showing the "四卡二日志"摘要与关键指标（system.json、stack.json、model.json、run.json、system.log、run.log）
- Upload form to submit `system.json`, `stack.json`, `model.json`, `run.json`, `system.log`, `run.log`, `energy_summary.json` and store them
- Admin summary page
- Auto-seed demo data on first run

Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

- 管理员登录地址: /admin/login
- 账号/密码: admin / admin123

需要更改的话，编辑 `openaiperf_site/app/main.py` 中 `admin_login` 函数里的用户名和密码即可。

Upload
- Go to `/submit` and upload up to five JSON files. The server will parse, store to `storage/<id>/` and index a summary to SQLite.

Notes
- This is an MVP. Schema fields are stored as JSON text as well as a few indexed columns for filtering/sorting. It is easy to extend the `Submission` model to add more structured fields.
- Tailwind is loaded via CDN for simplicity. Switch to a local build when needed.


