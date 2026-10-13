"""
app.py
------
Standalone Python Web Application Launcher for Viral Escape Predictor.
Runs a zero-dependency local web server and opens the interactive dashboard in your browser.

Usage:
    python app.py
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

PORT = 8000
DIRECTORY = Path(__file__).resolve().parent / "reports"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

def main():
    if not DIRECTORY.exists():
        print(f"Error: Reports directory {DIRECTORY} not found!")
        sys.exit(1)

    os.chdir(DIRECTORY)
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/index.html"
        print("="*70)
        print("  🧬 VIRAL MUTATION & IMMUNE ESCAPE PREDICTOR — USER WEB APP")
        print("="*70)
        print(f"  Local Web Server Running at: {url}")
        print("  Opening in your default browser...")
        print("  Press Ctrl+C to stop server.")
        print("="*70)

        webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server gracefully.")
            httpd.server_close()

if __name__ == "__main__":
    main()
