"""Dịch vụ dự đoán: kiểm tra dữ liệu -> tiền xử lý -> GRU (eGFR tương lai) -> K-Means (nhóm tiến triển) -> kết luận.

Toàn bộ logic tính toán bám sát GRU.ipynb (ô 6, 8, 9). Không huấn luyện lại, không tạo kết quả giả:
nếu thiếu tệp mô hình thì `Predictor.ready` là False và `run_pipeline` ném ModelNotReady.

Chạy bằng ONNX Runtime + NumPy (không cần PyTorch/scikit-learn) để vừa máy chủ nhỏ. Các tệp gru.onnx,
gru.json, phan_cum.json được xuất từ mô hình gốc bằng scripts/export_runtime_assets.py, script này kiểm tra
kết quả trùng với PyTorch/scikit-learn trước khi ghi.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

MONTHS_HISTORY = [0, 6, 12]
MONTHS_PREDICT = [18, 24, 30, 36]
THOI_GIAN = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])  # năm, ứng với 0M..36M
CONFIDENCE = {1: "Thấp", 2: "Trung bình", 3: "Cao"}
REQUIRED_FILES = ("gru.onnx", "gru.json", "phan_cum.json", "tham_so_tien_xu_ly.json")

# Mã gốc của CKD-ROUTE, đối chiếu với Bảng 2 của Iimori et al. 2018 (PLoS ONE): tỉ lệ nam 70,2%;
# nguyên nhân 7,7% / 59,7% / 6,8% / 25,9% ở nhóm đạm niệu bình thường (n = 352) khớp đúng với mã 1/2/3/4.
GENDER_NAMES = {1: "Nam", 2: "Nữ"}
ETIOLOGY_NAMES = {1: "Bệnh thận đái tháo đường", 2: "Xơ hoá thận", 3: "Viêm cầu thận", 4: "Nguyên nhân khác"}

# Tên hiển thị của 13 biến tĩnh và 5 đặc trưng phân cụm
STATIC_LABELS = {
    "age_z": "Tuổi (z-score)",
    "gender_2": "Giới tính = Nữ",
    "hypertension": "Tăng huyết áp",
    "diabetes": "Đái tháo đường",
    "prevalence of CVD": "Bệnh tim mạch",
    **{f"etiology_{k}": f"Nguyên nhân = {v}" for k, v in ETIOLOGY_NAMES.items()},
    "log1p_UPCR_z": "UPCR (log1p, z-score)",
    "UPCR_thieu": "Cờ thiếu UPCR",
    "Alb_z": "Albumin (z-score)",
    "SBP_z": "Huyết áp tâm thu (z-score)",
}
CLUSTER_FEATURE_LABELS = {
    "do_doc": "Độ dốc eGFR (mL/phút/1,73m²/năm)",
    "thay_doi_tong": "Thay đổi eGFR tới tháng 36",
    "log1p_UPCR_z": "UPCR (z-score)",
    "Alb_z": "Albumin (z-score)",
    "SBP_z": "Huyết áp tâm thu (z-score)",
}


class ModelNotReady(RuntimeError):
    """Thiếu tệp mô hình / bộ tiền xử lý."""


def ckd_stage(egfr: float) -> str:
    if egfr >= 90:
        return "G1"
    if egfr >= 60:
        return "G2"
    if egfr >= 45:
        return "G3a"
    if egfr >= 30:
        return "G3b"
    if egfr >= 15:
        return "G4"
    return "G5"


def slope_per_year(y: np.ndarray) -> float:
    """Độ dốc đường thẳng khớp nhất (theo năm) qua các điểm không NaN; NaN nếu < 2 điểm."""
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return float("nan")
    t = THOI_GIAN[ok]
    return float(((t - t.mean()) * (y[ok] - y[ok].mean())).sum() / ((t - t.mean()) ** 2).sum())


def last_observed(Vh: np.ndarray) -> np.ndarray:
    """eGFR (z) của lần đo thật gần nhất trong 0/6/12M; Vh (N,3) có NaN, cột 0 luôn có."""
    out = Vh[:, 0].copy()
    for t in (1, 2):
        out = np.where(np.isnan(Vh[:, t]), out, Vh[:, t])
    return out


def _r(v: float, d: int = 3) -> float | None:
    v = float(v)
    return None if math.isnan(v) else round(v, d)


class Predictor:
    def __init__(self, asset_dir: str | Path):
        self.asset_dir = Path(asset_dir)
        self.ready = False
        self.error: str | None = None
        self.version_info: dict[str, Any] = {}
        try:
            self._load()
            self.ready = True
        except Exception as exc:  # noqa: BLE001 - báo trạng thái cho /api/health thay vì làm sập app
            self.error = f"{type(exc).__name__}: {exc}"
            log.error("Không nạp được mô hình AI: %s", self.error)

    # ------------------------------------------------------------------ nạp
    def _load(self) -> None:
        missing = [f for f in REQUIRED_FILES if not (self.asset_dir / f).exists()]
        if missing:
            raise ModelNotReady(f"Thiếu tệp mô hình trong {self.asset_dir}: {', '.join(missing)}")

        self.params = json.loads((self.asset_dir / "tham_so_tien_xu_ly.json").read_text(encoding="utf-8"))
        self.static_cols: list[str] = self.params["cot_dau_vao_tinh"]
        g = json.loads((self.asset_dir / "gru.json").read_text(encoding="utf-8"))
        cfg = g["cau_hinh"]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.gru = ort.InferenceSession(str(self.asset_dir / "gru.onnx"), opts, providers=["CPUExecutionProvider"])
        self.n_seeds = int(g["n_seeds"])
        self.mu, self.sd = float(g["mu"]), float(g["sd"])

        pc = json.loads((self.asset_dir / "phan_cum.json").read_text(encoding="utf-8"))
        if pc["MO_HINH_CHON"] != "KMeans":
            raise ModelNotReady(f"Mô hình phân cụm được chọn là {pc['MO_HINH_CHON']}, dịch vụ này chỉ hỗ trợ KMeans.")
        self.cluster_cols: list[str] = pc["COT_VAO"]
        self.clip_lo, self.clip_hi = np.array(pc["can_duoi"]), np.array(pc["can_tren"])
        self.scaler_mean, self.scaler_scale = np.array(pc["scaler_mean"]), np.array(pc["scaler_scale"])
        self.centers = np.array(pc["tam_cum"])
        self.cluster_names: dict[int, str] = {int(k): v for k, v in pc["ten_km"].items()}
        self.bad_cluster = next(c for c, n in self.cluster_names.items() if n == "Chuyển biến xấu")
        self.fast_decline_threshold = float(pc["NGUONG_NHANH"])
        self.version_info = {
            "gru_config": cfg,
            "n_gru_seeds": self.n_seeds,
            "n_params_per_seed": int(g["n_params_per_seed"]),
            "runtime": "onnxruntime",
            "cluster_model": pc["MO_HINH_CHON"],
            "n_clusters": int(pc["K"]),
            "cluster_features": self.cluster_cols,
            "fast_decline_threshold": self.fast_decline_threshold,
        }

    # ------------------------------------------------------- tiền xử lý thô -> z
    def raw_to_model_inputs(self, raw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Chuyển dữ liệu lâm sàng thô sang đúng đầu vào mô hình.

        Trả về (Vh[3], S[13]) ở thang z-score, theo thứ tự cột trong tham_so_tien_xu_ly.json.
        Tham số z-score lấy từ tập Train (đúng như lúc huấn luyện).
        """
        z = self.params["zscore_lam_sang"]

        def zs(name: str, v: float) -> float:
            return (v - z[name]["mean"]) / z[name]["std"]

        upcr = raw.get("upcr")
        upcr_missing = upcr is None
        if upcr_missing:
            upcr = self.params["dien_thieu"]["UPCR"]  # giá trị điền thiếu cố định của tập Train
        etiology = int(raw["etiology"])
        static = {
            "age_z": zs("age", float(raw["age"])),
            "gender_2": 1.0 if int(raw["gender"]) == 2 else 0.0,
            "hypertension": float(bool(raw["hypertension"])),
            "diabetes": float(bool(raw["diabetes"])),
            "prevalence of CVD": float(bool(raw["cvd"])),
            "etiology_1": float(etiology == 1),
            "etiology_2": float(etiology == 2),
            "etiology_3": float(etiology == 3),
            "etiology_4": float(etiology == 4),
            "log1p_UPCR_z": zs("UPCR", math.log1p(float(upcr))),
            "UPCR_thieu": 1.0 if upcr_missing else 0.0,
            "Alb_z": zs("Alb", float(raw["albumin"])),
            "SBP_z": zs("SBP", float(raw["sbp"])),
        }
        S = np.array([static[c] for c in self.static_cols], dtype=np.float32)

        vh = np.full(3, np.nan, dtype=np.float32)
        for i, key in enumerate(("egfr_0m", "egfr_6m", "egfr_12m")):
            v = raw.get(key)
            if v is not None:
                vh[i] = (float(v) - self.mu) / self.sd
        if np.isnan(vh[0]):
            raise ValueError("Thiếu eGFR tại tháng 0 (bắt buộc).")
        return vh, S

    # ------------------------------------------------------------- lõi dự đoán
    def predict_arrays(self, Vh: np.ndarray, S: np.ndarray) -> dict[str, np.ndarray]:
        """Lõi dùng chung cho dự đoán thật và kiểm chứng offline. Vh (N,3) có NaN, S (N,13)."""
        if not self.ready:
            raise ModelNotReady(self.error or "Mô hình chưa sẵn sàng")
        per_seed = self._gru_per_seed(Vh, S)
        return self._cluster(Vh, S, per_seed.mean(0))

    def _gru_per_seed(self, Vh: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Dự đoán của từng seed GRU, đơn vị eGFR thật: (n_seed, N, 4). Trung bình theo seed = TapHop."""
        out = self.gru.run(None, {"Vh": np.asarray(Vh, dtype=np.float32), "S": np.asarray(S, dtype=np.float32)})[0]
        return out.astype(np.float64) * self.sd + self.mu

    def _cluster(self, Vh: np.ndarray, S: np.ndarray, du: np.ndarray) -> dict[str, np.ndarray]:
        """Từ eGFR đo (Vh) + eGFR GRU dự đoán (du) -> đặc trưng xu hướng -> K-Means."""
        Vh = np.asarray(Vh, dtype=np.float32)
        last = last_observed(Vh).astype(np.float64) * self.sd + self.mu
        ls = Vh.astype(np.float64) * self.sd + self.mu
        Y = np.concatenate([ls, du], axis=1)
        slope = np.array([slope_per_year(r) for r in Y])
        total = du[:, 3] - last
        # cột 2-4 của đầu vào phân cụm: log1p_UPCR_z, Alb_z, SBP_z
        col = {c: self.static_cols.index(c) for c in ("log1p_UPCR_z", "Alb_z", "SBP_z")}
        feats = {"do_doc": slope, "thay_doi_tong": total, **{c: S[:, i] for c, i in col.items()}}
        raw_feats = np.stack([np.asarray(feats[c], dtype=np.float64) for c in self.cluster_cols], 1)
        clipped = np.clip(raw_feats, self.clip_lo, self.clip_hi)             # cắt về phân vị 5–95% của Train
        X = (clipped - self.scaler_mean) / self.scaler_scale                 # StandardScaler (fit trên Train)
        dist = np.sqrt(((X[:, None, :] - self.centers[None]) ** 2).sum(-1))  # khoảng cách Euclid tới tâm cụm
        labels = dist.argmin(1)                                               # = KMeans.predict
        risk = -dist[:, self.bad_cluster]
        return {"du_doan": du, "lich_su": ls, "last": last, "do_doc": slope, "thay_doi_tong": total,
                "cluster_id": labels, "risk": risk, "features_raw": raw_feats, "features": clipped,
                "features_scaled": X, "distances": dist}

    # ------------------------------------------------- chạy đủ 5 bước cho 1 bệnh nhân
    def run_pipeline(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Chạy lần lượt: kiểm tra dữ liệu -> tiền xử lý -> GRU -> K-Means -> kết luận.

        `raw` đã qua kiểm tra kiểu/khoảng giá trị ở tầng API (pydantic). Mỗi bước trả về kết quả trung gian
        và thời gian chạy thực tế (ms) để giao diện hiển thị.
        """
        if not self.ready:
            raise ModelNotReady(self.error or "Mô hình chưa sẵn sàng")
        steps: list[dict[str, Any]] = []
        t0 = time.perf_counter()

        def done(key: str, title: str, detail: dict[str, Any], t_start: float) -> None:
            steps.append({"key": key, "title": title, "ms": round((time.perf_counter() - t_start) * 1000, 2),
                          "detail": detail})

        # ---- Bước 1: kiểm tra dữ liệu
        t = time.perf_counter()
        visits = [m for m, k in zip(MONTHS_HISTORY, ("egfr_0m", "egfr_6m", "egfr_12m")) if raw.get(k) is not None]
        n_visits = len(visits)
        checks = [
            {"status": "ok", "text": "Đủ các trường bắt buộc: eGFR tháng 0, tuổi, giới, nguyên nhân CKD, tiền sử, "
                                     "albumin, huyết áp tâm thu."},
            {"status": "ok", "text": "Mọi giá trị nằm trong khoảng hợp lệ."},
            {"status": "ok" if n_visits == 3 else "warn",
             "text": f"Số lần đo eGFR: {n_visits}/3 (tháng {', '.join(map(str, visits))})."
                     + ("" if n_visits == 3 else " Mốc thiếu được đánh dấu bằng cờ quan sát, mô hình vẫn dự đoán được.")},
            {"status": "ok" if raw.get("upcr") is not None else "warn",
             "text": "Có UPCR." if raw.get("upcr") is not None else
             f"Thiếu UPCR: dùng giá trị điền thiếu của tập Train ({self.params['dien_thieu']['UPCR']:.3f} g/g) "
             "và bật cờ UPCR_thieu = 1."},
        ]
        done("validate", "Kiểm tra dữ liệu", {"checks": checks, "n_visits": n_visits}, t)

        # ---- Bước 2: tiền xử lý
        t = time.perf_counter()
        Vh, S = self.raw_to_model_inputs(raw)
        z = self.params["zscore_lam_sang"]
        upcr_used = raw.get("upcr") if raw.get("upcr") is not None else self.params["dien_thieu"]["UPCR"]
        yes_no = lambda b: "có" if b else "không"  # noqa: E731
        raw_value = {
            "age_z": raw["age"], "gender_2": GENDER_NAMES.get(int(raw["gender"]), raw["gender"]),
            "hypertension": yes_no(raw["hypertension"]), "diabetes": yes_no(raw["diabetes"]),
            "prevalence of CVD": yes_no(raw["cvd"]),
            **{f"etiology_{i}": ETIOLOGY_NAMES.get(int(raw["etiology"]), raw["etiology"]) for i in (1, 2, 3, 4)},
            "log1p_UPCR_z": upcr_used, "UPCR_thieu": "đã đo UPCR" if raw.get("upcr") is not None else "chưa đo UPCR",
            "Alb_z": raw["albumin"], "SBP_z": raw["sbp"],
        }
        transform = {
            "age_z": f"(x − {z['age']['mean']:.2f}) / {z['age']['std']:.2f}",
            "log1p_UPCR_z": f"(ln(1+x) − {z['UPCR']['mean']:.3f}) / {z['UPCR']['std']:.3f}",
            "Alb_z": f"(x − {z['Alb']['mean']:.3f}) / {z['Alb']['std']:.3f}",
            "SBP_z": f"(x − {z['SBP']['mean']:.2f}) / {z['SBP']['std']:.2f}",
            "gender_2": "1 nếu Nữ", "UPCR_thieu": "1 nếu thiếu",
            **{f"etiology_{i}": "one-hot" for i in (1, 2, 3, 4)},
            "hypertension": "có = 1", "diabetes": "có = 1", "prevalence of CVD": "có = 1",
        }
        static_rows = [{"name": c, "label": STATIC_LABELS.get(c, c), "raw": raw_value.get(c),
                        "transform": transform.get(c, "0/1"), "value": _r(S[i], 4)}
                       for i, c in enumerate(self.static_cols)]
        seq_rows = [{"month": m, "egfr": raw.get(k), "observed": raw.get(k) is not None,
                     "z": _r(Vh[i], 4)}
                    for i, (m, k) in enumerate(zip(MONTHS_HISTORY, ("egfr_0m", "egfr_6m", "egfr_12m")))]
        done("preprocess", "Tiền xử lý", {
            "static": static_rows, "sequence": seq_rows,
            "egfr_zscore": f"(eGFR − {self.mu:.3f}) / {self.sd:.3f}",
            "note": "Tham số chuẩn hoá được tính trên tập Train lúc huấn luyện và lưu trong tham_so_tien_xu_ly.json.",
        }, t)

        # ---- Bước 3 + 4: GRU rồi K-Means (cùng lõi predict_arrays, tách thời gian từng phần)
        t = time.perf_counter()
        per_seed = self._gru_per_seed(Vh[None, :], S[None, :])[:, 0, :]   # (n_seed, 4)
        du = per_seed.mean(0)
        ls = Vh * self.sd + self.mu
        last = float(last_observed(Vh[None, :])[0]) * self.sd + self.mu
        done("gru", "GRU dự đoán eGFR", {
            "predicted": [{"month": m, "egfr": _r(du[i]), "stage": ckd_stage(float(du[i])),
                           "seed_min": _r(per_seed[:, i].min()), "seed_max": _r(per_seed[:, i].max())}
                          for i, m in enumerate(MONTHS_PREDICT)],
            "history": [{"month": m, "egfr": _r(ls[i])} for i, m in enumerate(MONTHS_HISTORY)],
            "last_observed": _r(last),
            "n_seeds": self.n_seeds,
            "config": self.version_info["gru_config"],
        }, t)

        t = time.perf_counter()
        r = self._cluster(Vh[None, :], S[None, :], du[None, :])
        cid = int(r["cluster_id"][0])
        cluster = self.cluster_names[cid]
        feats = [{"name": c, "label": CLUSTER_FEATURE_LABELS.get(c, c), "value": _r(r["features_raw"][0][i]),
                  "clipped": _r(r["features"][0][i]), "scaled": _r(r["features_scaled"][0][i])}
                 for i, c in enumerate(self.cluster_cols)]
        distances = sorted(({"cluster": self.cluster_names[k], "distance": _r(r["distances"][0][k])}
                            for k in self.cluster_names), key=lambda d: d["distance"])
        done("kmeans", "K-Means phân nhóm", {
            "features": feats, "distances": distances, "cluster": cluster,
            "risk_score": _r(r["risk"][0], 4),
            "slope_per_year": _r(r["do_doc"][0]), "total_change": _r(r["thay_doi_tong"][0]),
        }, t)

        # ---- Bước 5: kết luận + cảnh báo
        t = time.perf_counter()
        stage36 = ckd_stage(float(du[3]))
        warns: list[dict[str, str]] = []
        if cluster == "Chuyển biến xấu":
            warns.append({"level": "high", "text": "Nhóm “Chuyển biến xấu”: eGFR dự kiến giảm nhanh hơn phần lớn "
                                                   "bệnh nhân trong dữ liệu huấn luyện."})
        if stage36 in ("G4", "G5"):
            warns.append({"level": "high", "text": f"eGFR dự đoán ở tháng 36 thuộc {stage36} (dưới 30): "
                                                   "cần được nhân viên y tế đánh giá."})
        if n_visits == 1:
            warns.append({"level": "info", "text": "Chỉ có 1 lần đo eGFR: độ tin cậy thấp."})
        elif n_visits == 2:
            warns.append({"level": "info", "text": "Thiếu 1 lần đo eGFR: độ tin cậy ở mức trung bình."})
        if raw.get("upcr") is None:
            warns.append({"level": "info", "text": "Thiếu UPCR: dùng giá trị điền thiếu của tập huấn luyện."})
        done("conclude", "Kết quả và cảnh báo", {
            "cluster": cluster, "stage_36m": stage36, "egfr_36m": _r(du[3]), "last_observed": _r(last),
            "change_to_36m": _r(du[3] - last), "slope_per_year": _r(r["do_doc"][0]),
            "confidence": CONFIDENCE[n_visits], "warnings": warns,
        }, t)

        return {"steps": steps, "total_ms": round((time.perf_counter() - t0) * 1000, 2)}
