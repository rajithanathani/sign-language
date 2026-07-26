"""
Vercel Serverless Function Entrypoint for FastAPI Backend

This module exposes the FastAPI application instance `app` from `backend.app`
for Vercel Serverless Functions environment (`@vercel/python`).
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.app import app
