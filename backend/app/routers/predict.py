"""API duy nhất của hệ thống: kiểm tra trạng thái mô hình, bệnh nhân mẫu và chạy dự đoán."""
import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ..ml.predictor import ModelNotReady, Predictor
from ..schemas import PatientInput

router = APIRouter(prefix="/api", tags=["predict"])
DATA = Path(__file__).resolve().parents[1] / "data"


def _predictor(request: Request) -> Predictor:
    return request.app.state.predictor


@router.get("/health")
def health(request: Request):
    p = _predictor(request)
    return {"status": "ok" if p.ready else "model_not_ready", "model_ready": p.ready, "model_error": p.error,
            "model": p.version_info}


@lru_cache
def _test_patients() -> list:
    f = DATA / "test_patients.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


@router.get("/test-patients")
def test_patients():
    """171 bệnh nhân tập Test (giá trị gốc từ CKD-ROUTE) để chọn thử, kèm kết quả notebook và eGFR thật để đối chiếu."""
    return _test_patients()


@router.post("/predict")
def predict(body: PatientInput, request: Request):
    p = _predictor(request)
    try:
        out = p.run_pipeline(body.model_dump())
    except ModelNotReady as exc:
        raise HTTPException(503, {"error": "model_not_ready", "message": f"Mô hình chưa sẵn sàng: {exc}"}) from exc
    except ValueError as exc:
        raise HTTPException(422, {"error": "invalid_input", "message": str(exc)}) from exc
    return {"input": body.model_dump(), **out}
