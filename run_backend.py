"""Script to run the backend server."""

import uvicorn
from backend.config import Config

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=True,
        log_level="info"
    )

