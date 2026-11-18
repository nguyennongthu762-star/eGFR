"""Tạo app/data/test_patients.json: toàn bộ 171 bệnh nhân tập TEST để chọn trên giao diện.

Giá trị nhập lấy NGUYÊN GỐC từ ROUTE_proteinuria_dataset.xlsx theo ID (không đổi ngược từ z-score),
làm tròn như dữ liệu gốc. Ô thiếu SBP/Albumin được điền bằng đúng giá trị điền thiếu của tập Train
(như lúc huấn luyện) và ghi vào "filled"; UPCR và eGFR tháng 6/12 thiếu thì để trống (mô hình có cờ riêng).
Kèm dự đoán của notebook (ket_qua_toan_he_thong_test.csv) và eGFR thật 18–36 tháng để đối chiếu.

Usage: python scripts/build_test_patients.py <thu_muc_Demo_HM> <ROUTE_proteinuria_dataset.xlsx>
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
demo = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT.parent / "Demo_HM")
raw_path = Path(sys.argv[2] if len(sys.argv) > 2 else Path.home() / "Downloads" / "ROUTE_proteinuria_dataset.xlsx")
P = json.loads((ROOT / "ml_assets" / "tham_so_tien_xu_ly.json").read_text(encoding="utf-8"))
FILL = P["dien_thieu"]
M_PRED = [18, 24, 30, 36]

test = pd.read_excel(demo / "test.xlsx").set_index("ID")
raw = pd.read_excel(raw_path).set_index("ID").loc[test.index]
ref = pd.read_csv(demo / "ket_qua_toan_he_thong_test.csv").set_index("ID").loc[test.index]


def val(v, nd):
    return None if pd.isna(v) else round(float(v), nd)


out = []
for i, r in raw.iterrows():
    filled = []
    sbp, alb = val(r["SBP"], 0), val(r["Alb"], 1)
    if sbp is None:
        sbp, filled = float(FILL["SBP"]), filled + ["sbp"]
    if alb is None:
        alb, filled = float(FILL["Alb"]), filled + ["albumin"]
    row = ref.loc[i]
    out.append({
        "id": int(i),
        "input": {
            "egfr_0m": val(r["eGFR(0M)"], 2), "egfr_6m": val(r["eGFR(6M)"], 2), "egfr_12m": val(r["eGFR(12M)"], 2),
            "age": int(r["age"]), "gender": int(r["gender"]), "etiology": int(r["etiology of CKD"]),
            "hypertension": bool(r["hypertension"]), "diabetes": bool(r["diabetes"]),
            "cvd": bool(r["prevalence of CVD"]),
            "sbp": sbp, "albumin": alb, "upcr": val(r["UPCR"], 3),
        },
        "filled": filled,
        "notebook": {"cluster": row["nhom"], "predicted": {m: round(float(row[f"du_doan_{m}M"]), 2) for m in M_PRED}},
        "actual": {m: val(row[f"THAT_{m}M"], 2) for m in M_PRED},
    })

dst = ROOT / "app" / "data" / "test_patients.json"
dst.parent.mkdir(exist_ok=True)
dst.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"Đã ghi {len(out)} bệnh nhân -> {dst}")
print("Điền thiếu:", {p["id"]: p["filled"] for p in out if p["filled"]})
