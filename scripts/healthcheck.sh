#!/bin/sh
python -c "
import httpx
try:
    r = httpx.get('http://localhost:5050/health', timeout=5)
    r.raise_for_status()
except Exception:
    exit(1)
"
