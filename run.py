#!/usr/bin/env python3
"""Launch WanForge.

Runs as an NSSM service, so auto-reload is OFF by default. Uvicorn's
--reload on Windows spawns a worker child per-reload; under a service
manager a reload cycle can orphan the worker holding the listening
socket, silently wedging the port. Set GENAI_RELOAD=1 for local dev.
"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=os.environ.get("GENAI_RELOAD", "0") == "1",
        log_level="info",
    )
