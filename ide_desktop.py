# -*- coding: utf-8 -*-
"""One-Code Desktop IDE — PyWebView 桌面客户端"""
import os, sys, io, threading

# Suppress all console output
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

IDE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("ONE_CODE_PORT", "8765"))


def start_server():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    from ide_server import app
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)


def main():
    # Start Flask in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait briefly for server to be ready
    import time
    import urllib.request
    for _ in range(30):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/status", timeout=1)
            break
        except Exception:
            time.sleep(0.3)

    try:
        import webview
        webview.create_window(
            "One-Code IDE",
            f"http://127.0.0.1:{PORT}",
            width=1300,
            height=850,
            min_size=(900, 600),
            text_select=True,
        )
        webview.start()
    except ImportError:
        print("PyWebView not installed. Opening in browser...")
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{PORT}")
        print(f"IDE running at http://127.0.0.1:{PORT}")
        print("Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
