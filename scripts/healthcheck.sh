#!/bin/sh
python -c "
import httpx
import os
port = os.environ.get('PORT', '5050')
try:
    r = httpx.get(f'http://localhost:{port}/health', timeout=5)
    r.raise_for_status()
except Exception:
    exit(1)
"
