import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .ml.predictor import Predictor
from .routers import predict

settings = get_settings()
logging.basicConfig(level=logging.INFO)

FIELD_LABELS = {
    "egfr_0m": "eGFR tháng 0", "egfr_6m": "eGFR tháng 6", "egfr_12m": "eGFR tháng 12", "age": "Tuổi",
    "gender": "Giới tính", "etiology": "Nguyên nhân CKD", "hypertension": "Tăng huyết áp",
    "diabetes": "Đái tháo đường", "cvd": "Bệnh tim mạch", "sbp": "Huyết áp tâm thu", "albumin": "Albumin",
    "upcr": "UPCR",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor(settings.model_asset_dir)  # nạp mô hình một lần khi khởi động
    yield


app = FastAPI(title="CKD eGFR – GRU + K-Means", version="2.0.0", lifespan=lifespan,
              docs_url="/api/docs" if settings.environment != "production" else None,
              redoc_url=None, openapi_url="/api/openapi.json" if settings.environment != "production" else None)

app.add_middleware(CORSMiddleware, allow_origins=settings.cors_list, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


# Định dạng lỗi thống nhất: {"error": <mã>, "message": <thông điệp>, "details"?: [...]}
@app.exception_handler(HTTPException)
async def http_exc(_: Request, exc: HTTPException):
    body = exc.detail if isinstance(exc.detail, dict) else {"error": "error", "message": str(exc.detail)}
    return JSONResponse(body, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exc(_: Request, exc: RequestValidationError):
    details = []
    for e in exc.errors():
        field = ".".join(str(p) for p in e["loc"][1:])
        details.append({"field": field, "label": FIELD_LABELS.get(field, field), "message": e["msg"]})
    return JSONResponse({"error": "validation_error", "message": "Dữ liệu không hợp lệ", "details": details},
                        status_code=422)


app.include_router(predict.router)
