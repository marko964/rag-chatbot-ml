from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from pathlib import Path

from app.config import settings
from app.db.database import create_tables
from app.limiter import limiter
from app.rag.ingest import ingest_directory
from app.rag.retriever import _get_collection
from app.api import chat, admin


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    # Auto-ingest documents on startup if collection is empty
    try:
        col = _get_collection()
        if col.count() == 0:
            docs_dir = Path("/app/kb_docs")
            if docs_dir.exists():
                ingest_directory(docs_dir)
    except Exception:
        pass
    yield


app = FastAPI(
    title=f"{settings.company_name} Chatbot API",
    lifespan=lifespan,
    # Hide docs in production to reduce attack surface
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "X-Api-Key"],
)

app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(admin.router, prefix="/api/v1", tags=["admin"])


@app.middleware("http")
async def no_cache(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
