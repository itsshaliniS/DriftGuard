from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from .routes import router

app = FastAPI(title="Drift Guard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

