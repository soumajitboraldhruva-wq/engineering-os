from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from api import calculator, matrices, ode, graphing, analyser

app = FastAPI(title="Engineering OS API")

# CORS: allow local dev + deployed frontend (set via ALLOWED_ORIGINS env var)
# Format: "https://your-app.vercel.app,https://another-domain.com"
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins if o.strip()] + default_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    calculator.router,
    prefix="/api/calc",
    tags=["Calculator"],
)

app.include_router(
    matrices.router,
    prefix="/api/matrix",
    tags=["Linear Algebra"],
)

app.include_router(
    ode.router,
    prefix="/api/ode",
    tags=["Dynamics"],
)

app.include_router(
    graphing.router,
    prefix="/api/graph",
    tags=["Graphing"],
)

app.include_router(
    analyser.router,
    prefix="/api/analyser",
    tags=["Sequence & Curve Analyser"],
)

# The Cube Solver is switched off for now (hidden, not deleted). Set ENABLE_CUBE_SOLVER=1 to bring it back
# (also set VITE_SHOW_CUBE_SOLVER=true for the frontend).
if os.getenv("ENABLE_CUBE_SOLVER", "0").strip().lower() in ("1", "true", "yes", "on"):
    from api import cube

    app.include_router(
        cube.router,
        prefix="/api/cube",
        tags=["Cube Solver"],
    )


@app.get("/")
def read_root():
    return {"message": "Engineering OS Backend is running"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
