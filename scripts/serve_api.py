"""Launch the FastAPI service locally.

Usage:
    python scripts/serve_api.py
"""

import os
import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "graphsage.api.app:app",
        host=os.getenv("GRAPHSAGE_API_HOST", "0.0.0.0"),
        port=int(os.getenv("GRAPHSAGE_API_PORT", "8000")),
        reload=True,
    )
