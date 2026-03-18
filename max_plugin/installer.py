"""
installer.py
------------
Silent dependency installer and Ollama manager.
Called once at plugin startup — no terminal, no user intervention needed.

Responsibilities
----------------
1. Install missing Python packages into 3ds Max's own Python interpreter.
2. Detect / start / query Ollama (the free local LLM backend).
3. Pull Ollama models on demand.

All network/subprocess errors are caught and returned as (ok, message) tuples
so the MAXScript UI can display them without crashing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

# ---------------------------------------------------------------------------
# Required pip packages
# ---------------------------------------------------------------------------
REQUIRED_PACKAGES: list[str] = ["openai"]

# ---------------------------------------------------------------------------
# Ollama constants
# ---------------------------------------------------------------------------
OLLAMA_API_BASE   = "http://localhost:11434"
OLLAMA_MODELS_URL = f"{OLLAMA_API_BASE}/api/tags"
OLLAMA_VER_URL    = f"{OLLAMA_API_BASE}/api/version"
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"

# Curated model recommendations: (model_tag, display_label)
RECOMMENDED_MODELS: list[tuple[str, str]] = [
    ("llama3.2:3b",  "Llama 3.2 3B  – fast, ~2 GB"),
    ("mistral:7b",   "Mistral 7B    – balanced, ~4 GB"),
    ("qwen2.5:7b",   "Qwen 2.5 7B   – best quality, ~4 GB"),
]


# ===========================================================================
# Pip dependency installer
# ===========================================================================

def ensure_dependencies() -> tuple[bool, str]:
    """
    Check for and silently install missing packages into Max's Python.

    Returns (success: bool, message: str).
    """
    missing = [pkg for pkg in REQUIRED_PACKAGES if not _importable(pkg)]
    if not missing:
        return True, "Dependencies OK."

    errors: list[str] = []
    for pkg in missing:
        ok, msg = _pip_install(pkg)
        if not ok:
            errors.append(msg)

    if errors:
        return False, "Install failed: " + "; ".join(errors)
    return True, f"Installed: {', '.join(missing)}"


def _importable(package_name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(package_name) is not None


def _pip_install(package_name: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name, "-q", "--user"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return False, result.stderr.strip()[:200]
        return True, f"{package_name} installed."
    except subprocess.TimeoutExpired:
        return False, f"Timed out installing {package_name}."
    except Exception as exc:
        return False, str(exc)[:200]


# ===========================================================================
# Ollama helpers
# ===========================================================================

def ollama_status() -> tuple[bool, bool]:
    """
    Return (is_running, is_installed).

    is_running  – the Ollama HTTP server is accepting connections
    is_installed – the ollama binary exists on this machine
    """
    running = _ollama_reachable()
    if running:
        return True, True
    installed = bool(_find_ollama_exe())
    return False, installed


def start_ollama() -> tuple[bool, str]:
    """
    Launch the Ollama server in the background.
    Returns (success, message).
    """
    exe = _find_ollama_exe()
    if not exe:
        return False, "Ollama binary not found. Please install Ollama first."

    try:
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        subprocess.Popen(
            [exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        # Wait up to 8 s for the server to come up
        for _ in range(8):
            time.sleep(1)
            if _ollama_reachable():
                return True, "Ollama started."
        return False, "Ollama launched but did not respond in time. Try again."
    except Exception as exc:
        return False, str(exc)[:200]


def list_local_models() -> list[str]:
    """Return names of Ollama models already downloaded on this machine."""
    if not _ollama_reachable():
        return []
    try:
        with urllib.request.urlopen(OLLAMA_MODELS_URL, timeout=4) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def pull_model(model_tag: str, progress_callback=None) -> tuple[bool, str]:
    """
    Pull an Ollama model.  This blocks until the download completes.

    progress_callback(pct: int, status: str) – optional; called periodically.
    Returns (success, message).
    """
    exe = _find_ollama_exe()
    if not exe:
        return False, "Ollama binary not found."
    try:
        proc = subprocess.Popen(
            [exe, "pull", model_tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last_line = ""
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.strip()
            if line:
                last_line = line
                if progress_callback:
                    # Extract percentage if present
                    pct = _extract_pct(line)
                    progress_callback(pct, line[:80])
        proc.wait(timeout=600)
        if proc.returncode == 0:
            return True, f"Model '{model_tag}' is ready."
        return False, f"Pull failed: {last_line[:200]}"
    except subprocess.TimeoutExpired:
        return False, "Model download timed out (10 min limit)."
    except Exception as exc:
        return False, str(exc)[:200]


def open_ollama_download_page() -> None:
    """Open the Ollama download page in the system browser."""
    import webbrowser
    webbrowser.open(OLLAMA_DOWNLOAD_URL)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ollama_reachable() -> bool:
    try:
        urllib.request.urlopen(OLLAMA_VER_URL, timeout=2)
        return True
    except Exception:
        return False


def _find_ollama_exe() -> Optional[str]:
    # PATH lookup
    exe = shutil.which("ollama")
    if exe:
        return exe
    # Windows common install locations
    candidates = [
        os.path.expanduser(r"~\AppData\Local\Programs\Ollama\ollama.exe"),
        r"C:\Program Files\Ollama\ollama.exe",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _extract_pct(line: str) -> int:
    """Try to parse a percentage from an Ollama progress line like '42%'."""
    import re
    m = re.search(r"(\d+)%", line)
    return int(m.group(1)) if m else -1
