import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DEMO = Path(__file__).resolve().parents[2] / "Demo_HM"

BASE = {"egfr_0m": 45.5, "egfr_6m": 43.0, "egfr_12m": 45.3, "age": 65, "gender": 1, "etiology": 2,
        "hypertension": True, "diabetes": False, "cvd": False, "sbp": 140, "albumin": 3.8, "upcr": 0.9}
STEP_KEYS = ["validate", "preprocess", "gru", "kmeans", "conclude"]


def test_health_model_ready(client):
    r = client.get("/api/health").json()
    assert r["model_ready"] is True, r
    assert r["model"]["cluster_model"] == "KMeans" and r["model"]["n_gru_seeds"] == 5


def test_predict_runs_all_five_steps(client):
    r = client.post("/api/predict", json=BASE)
    assert r.status_code == 200, r.text
    d = r.json()
    assert [s["key"] for s in d["steps"]] == STEP_KEYS
    st = {s["key"]: s["detail"] for s in d["steps"]}
    assert len(st["preprocess"]["static"]) == 13
    assert [p["month"] for p in st["gru"]["predicted"]] == [18, 24, 30, 36]
    assert all(5 <= p["egfr"] <= 150 for p in st["gru"]["predicted"])
    assert st["kmeans"]["cluster"] in ("Chuyển biến xấu", "Ổn định", "Cải thiện")
    assert st["kmeans"]["distances"][0]["cluster"] == st["kmeans"]["cluster"]  # nhóm = tâm cụm gần nhất
    assert st["conclude"]["confidence"] == "Cao"


def test_missing_visits_and_upcr_lower_confidence(client):
    d = client.post("/api/predict", json=dict(BASE, egfr_6m=None, egfr_12m=None, upcr=None)).json()
    st = {s["key"]: s["detail"] for s in d["steps"]}
    assert st["conclude"]["confidence"] == "Thấp"
    assert st["validate"]["n_visits"] == 1
    upcr_flag = next(r for r in st["preprocess"]["static"] if r["name"] == "UPCR_thieu")
    assert upcr_flag["value"] == 1.0
    assert any("UPCR" in w["text"] for w in st["conclude"]["warnings"])


@pytest.mark.parametrize("bad", [
    {"egfr_0m": None},          # thiếu eGFR tháng 0
    {"sbp": 900},               # ngoài khoảng
    {"egfr_0m": -5},
    {"etiology": 7},
    {"albumin": "abc"},
])
def test_invalid_input_rejected(client, bad):
    r = client.post("/api/predict", json=dict(BASE, **bad))
    assert r.status_code == 422
    assert r.json()["error"] == "validation_error" and r.json()["details"]


def test_all_test_patients_reproduce_notebook(client):
    """Cả 171 bệnh nhân tập Test (giá trị gốc CKD-ROUTE) chạy qua API phải cho cùng kết quả với notebook."""
    pts = client.get("/api/test-patients").json()
    assert len(pts) == 171
    for e in pts:
        st = {s["key"]: s["detail"] for s in client.post("/api/predict", json=e["input"]).json()["steps"]}
        got = {p["month"]: p["egfr"] for p in st["gru"]["predicted"]}
        for m, v in e["notebook"]["predicted"].items():
            assert abs(got[int(m)] - v) < 0.05, (e["id"], m, got[int(m)], v)
        assert st["kmeans"]["cluster"] == e["notebook"]["cluster"], e["id"]


def test_predict_matches_notebook_reference(client):
    """BN 10208 (tập Test), dữ liệu thô tái dựng từ z-score -> phải khớp ket_qua_toan_he_thong_test.csv."""
    if not (DEMO / "test.xlsx").exists():
        pytest.skip("không có thư mục Demo_HM")
    row = pd.read_excel(DEMO / "test.xlsx").query("ID == 10208").iloc[0]
    z = json.load(open(DEMO / "tham_so_tien_xu_ly.json", encoding="utf-8"))["zscore_lam_sang"]
    mu, sd = 33.32484592307998, 18.339739229258125
    body = {"age": float(row.age_z * z["age"]["std"] + z["age"]["mean"]),
            "sbp": float(row.SBP_z * z["SBP"]["std"] + z["SBP"]["mean"]),
            "albumin": float(row.Alb_z * z["Alb"]["std"] + z["Alb"]["mean"]),
            "egfr_0m": float(row.eGFR_0M_in * sd + mu), "egfr_6m": float(row.eGFR_6M_in * sd + mu),
            "egfr_12m": float(row.eGFR_12M_in * sd + mu),
            "gender": 2 if row.gender_2 == 1 else 1,
            "etiology": int([i for i in (1, 2, 3, 4) if row[f"etiology_{i}"] == 1][0]),
            "hypertension": bool(row.hypertension), "diabetes": bool(row.diabetes),
            "cvd": bool(row["prevalence of CVD"]),
            "upcr": None if row.UPCR_thieu == 1 else float(np.expm1(row.log1p_UPCR_z * z["UPCR"]["std"] + z["UPCR"]["mean"]))}
    st = {s["key"]: s["detail"] for s in client.post("/api/predict", json=body).json()["steps"]}
    ref = pd.read_csv(DEMO / "ket_qua_toan_he_thong_test.csv").query("ID == 10208").iloc[0]
    got = {p["month"]: p["egfr"] for p in st["gru"]["predicted"]}
    for m in (18, 24, 30, 36):
        assert abs(got[m] - ref[f"du_doan_{m}M"]) < 0.01, (m, got[m], ref[f"du_doan_{m}M"])
    assert st["kmeans"]["cluster"] == ref["nhom"]
