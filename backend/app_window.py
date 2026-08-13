"""
PatchDentalNet DSS - Desktop App Launcher
==========================================
Runs the FastAPI backend in a background thread and displays it in a native
app window via pywebview (no address bar / browser tabs - looks and feels
like a real desktop app). This is what the desktop shortcut icon launches.

A branded loading screen is shown immediately; the actual app UI is swapped
in once the backend (CNN models, ~900MB) has finished loading, which can
take up to ~30 seconds on first launch.

Run directly for testing:  pythonw.exe app_window.py   (no console window)
                            python.exe app_window.py    (console window, for debugging)
"""

import os
import sys
import threading
import time
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import uvicorn
import webview

HOST = "127.0.0.1"
PORT = 8001

LOADING_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>PatchDentalNet DSS</title>
<style>
  html, body { height: 100%; margin: 0; }
  body {
    display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 18px;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: linear-gradient(160deg, #2a78d6 0%, #184f95 100%);
    color: #fff;
  }
  .logo { width: 64px; height: 64px; }
  .title { font-size: 1.3rem; font-weight: 800; letter-spacing: -0.01em; }
  .sub { font-size: 0.85rem; opacity: 0.85; max-width: 320px; text-align: center; line-height: 1.5; }
  .spinner {
    width: 30px; height: 30px; border: 3px solid rgba(255,255,255,0.3); border-top-color: #fff;
    border-radius: 50%; animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style></head>
<body>
  <svg class="logo" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <rect x="1" y="1" width="46" height="46" rx="13" fill="rgba(255,255,255,0.16)"/>
    <path d="M24 13c-3.2 0-5.1 1.7-6.9 1.7-1.6 0-2.8-1-4.3-1-2.6 0-5.3 2.2-5.3 6.9 0 4.3 3.1 12.4 5.6 12.4 1.5 0 2-1 3.3-1 1.3 0 1.6 1 3.3 1 1.1 0 1.6-1.1 2.1-2.4.2-.6 1.1-.6 1.4 0 .5 1.3 1 2.4 2.1 2.4 1.7 0 2-1 3.3-1 1.3 0 1.8 1 3.3 1 2.5 0 5.6-8.1 5.6-12.4 0-4.7-2.7-6.9-5.3-6.9-1.5 0-2.7 1-4.3 1-1.8 0-3.7-1.7-6.9-1.7Z" fill="#fff"/>
    <circle cx="24" cy="23" r="2.1" fill="#2a78d6"/>
  </svg>
  <div class="title">PatchDentalNet DSS</div>
  <div class="spinner"></div>
  <div class="sub">Loading AI models — this can take up to 30 seconds on first launch.</div>
</body></html>"""


def _wait_for_server(url: str, timeout: float = 90.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=1.5)
            return True
        except Exception:
            time.sleep(0.4)
    return False


def _startup(window):
    """Runs in a background thread once the pywebview event loop has started
    (loading screen is already visible at this point)."""
    import main as backend_main  # heavy: loads CNN models + SVM pipeline (~15-25s)

    server_thread = threading.Thread(
        target=lambda: uvicorn.run(backend_main.app, host=HOST, port=PORT, log_level="warning"),
        daemon=True,
    )
    server_thread.start()

    app_url = f"http://{HOST}:{PORT}"
    if _wait_for_server(app_url + "/api/health"):
        window.load_url(app_url)
    else:
        window.load_html("<h2 style='font-family:sans-serif;color:#d03b3b'>Backend failed to start. "
                          "Check the console output for errors.</h2>")


if __name__ == "__main__":
    # Disabled by default in pywebview - without this, the "Download Report
    # (HTML)" / "Download Output (JSON)" buttons silently do nothing (the
    # blob: URL + <a download> click never triggers a save dialog).
    webview.settings["ALLOW_DOWNLOADS"] = True

    window = webview.create_window(
        "PatchDentalNet DSS",
        html=LOADING_HTML,
        width=1360,
        height=860,
        min_size=(1000, 650),
    )
    webview.start(_startup, window)
