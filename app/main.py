from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Sales Forecasting API")
app.include_router(router)

