"""全局配置与本地持久化设置。"""
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent          # backend/
PROJECT_DIR = BASE_DIR.parent                               # 项目根
FRONTEND_DIR = PROJECT_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "chat.db"
SETTINGS_PATH = DATA_DIR / "settings.json"

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_LLM_BASE = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_TOP_K = 4

DEFAULT_SETTINGS = {
    "llm_base_url": DEFAULT_LLM_BASE,
    "llm_api_key": "",
    "llm_model": DEFAULT_LLM_MODEL,
    "top_k": DEFAULT_TOP_K,
}


def ensure_dirs():
    for d in (DATA_DIR, UPLOAD_DIR, CHROMA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict:
    ensure_dirs()
    s = dict(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        try:
            s.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return s


def save_settings(new: dict):
    ensure_dirs()
    s = load_settings()
    for k in ("llm_base_url", "llm_api_key", "llm_model", "top_k"):
        if k in new and new[k] is not None:
            s[k] = new[k]
    SETTINGS_PATH.write_text(
        json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return s


def public_settings() -> dict:
    """返回给前端的设置（永不回显 api_key）。"""
    s = load_settings()
    return {
        "has_llm": bool(s.get("llm_api_key", "").strip()),
        "llm_base_url": s.get("llm_base_url", DEFAULT_LLM_BASE),
        "llm_model": s.get("llm_model", DEFAULT_LLM_MODEL),
        "top_k": int(s.get("top_k", DEFAULT_TOP_K)),
    }
