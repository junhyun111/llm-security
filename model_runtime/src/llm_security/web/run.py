from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "llm_security.web.app:app",
        host=os.getenv("WEB_HOST", "127.0.0.1"),
        port=int(os.getenv("WEB_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
