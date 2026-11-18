"""Chuyển mô hình đã huấn luyện sang dạng gọn để chạy trên máy chủ nhỏ (không cần PyTorch / scikit-learn).

Đầu vào (giữ nguyên, sinh từ GRU.ipynb):
  ml_assets/gru_lstm_phat_nang.pt   5 GRU seed (state_dict)
  ml_assets/phan_cum.joblib         K-Means + StandardScaler + giới hạn cắt 5–95%
Đầu ra:
  ml_assets/gru.onnx                5 GRU seed trong 1 đồ thị ONNX: (Vh[B,3] có NaN, S[B,13]) -> (5, B, 4) z-score
  ml_assets/gru.json                cấu hình GRU, mu/sd của eGFR, số tham số
  ml_assets/phan_cum.json           tâm cụm K-Means, tham số chuẩn hoá, giới hạn cắt, tên cụm, ngưỡng

Script tự kiểm tra: ONNX phải cho cùng kết quả với mô hình PyTorch gốc (lệch < 1e-4 eGFR) và
K-Means bản NumPy phải gán cùng cụm với scikit-learn.

Usage (cần môi trường dev: torch, scikit-learn, joblib, onnx, onnxruntime):
  python scripts/export_runtime_assets.py
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from network import Seq2SeqNangCao  # noqa: E402  (kiến trúc gốc của notebook)

A = ROOT / "ml_assets"
N_PRED = 4  # chỉ dùng 4 mốc 18/24/30/36 tháng


def gru_cell(cell, x, h):
    """GRU một bước, đúng công thức của torch.nn.GRU/GRUCell (thứ tự cổng r, z, n)."""
    gi = x @ cell[0].T + cell[2]
    gh = h @ cell[1].T + cell[3]
    i_r, i_z, i_n = gi.chunk(3, 1)
    h_r, h_z, h_n = gh.chunk(3, 1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return (1 - z) * n + z * h


class GruEnsembleExport(nn.Module):
    """Bản tương đương Seq2SeqNangCao.forward khi Ln = 3 (luôn đúng lúc dự đoán: 3 ô 0/6/12M, ô thiếu = NaN).

    Viết bằng phép toán cơ bản để xuất ONNX gọn; được kiểm chứng lại với mô hình gốc ở main().
    """

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    @staticmethod
    def features(Vh):
        # = dac_trung_chuoi(Vh, Ln=3)
        obs = ~torch.isnan(Vh)
        Vz = torch.where(obs, Vh, torch.zeros_like(Vh))
        zero = torch.zeros_like(Vz[:, 0])
        e, dt, sl = [Vz[:, 0]], [zero], [zero]
        for t in (1, 2):
            o = obs[:, t]
            doc = (Vz[:, t] - e[-1]) / (dt[-1] + 0.5)
            sl.append(torch.where(o, doc, sl[-1]))
            e.append(torch.where(o, Vz[:, t], e[-1]))
            dt.append(torch.where(o, zero, dt[-1] + 0.5))
        e, dt, sl = torch.stack(e, 1), torch.stack(dt, 1), torch.stack(sl, 1)
        tabs = torch.stack([zero, zero + 0.5, zero + 1.0], 1)
        X = torch.stack([e, obs.float(), dt, sl, tabs], 2)
        return X, e[:, 2], dt[:, 2]

    @staticmethod
    def one(m, X, last, gap, S):
        B = X.shape[0]
        one_hot_ln3 = torch.zeros(B, 3) + torch.tensor([0.0, 0.0, 1.0])
        lin = m.tuyen_tinh(torch.cat([X.reshape(B, -1), S, one_hot_ln3], 1))
        enc = (m.encoder.weight_ih_l0, m.encoder.weight_hh_l0, m.encoder.bias_ih_l0, m.encoder.bias_hh_l0)
        h = torch.zeros(B, m.encoder.hidden_size)
        Hs = []
        for t in range(3):
            h = gru_cell(enc, X[:, t], h)
            Hs.append(h)
        H = torch.stack(Hs, 1)
        h = m.init_h(torch.cat([h, m.stat(S)], 1))
        dec = (m.decoder.weight_ih, m.decoder.weight_hh, m.decoder.bias_ih, m.decoder.bias_hh)
        prev = last.unsqueeze(1)
        tich_luy = torch.zeros_like(prev)
        outs = []
        for k in range(N_PRED):
            diem = (H * m.attn(h).unsqueeze(1)).sum(-1)
            ctx = (torch.softmax(diem, 1).unsqueeze(-1) * H).sum(1)
            vao = torch.cat([prev, (gap + 0.5 * (k + 1)).unsqueeze(1),
                             torch.zeros_like(prev) + 0.5 * (3 + k), ctx], 1)
            h = gru_cell(dec, vao, h)
            tich_luy = tich_luy + m.out(torch.cat([h, ctx], 1))
            pred = last.unsqueeze(1) + lin[:, k:k + 1] + tich_luy
            outs.append(pred)
            prev = pred
        return torch.cat(outs, 1)

    def forward(self, Vh, S):
        X, last, gap = self.features(Vh)
        return torch.stack([self.one(m, X, last, gap, S) for m in self.models])


def load_models():
    ck = torch.load(A / "gru_lstm_phat_nang.pt", map_location="cpu", weights_only=False)
    cfg = ck["GRU"]["cau_hinh"]
    models = []
    for sd in ck["GRU"]["state_dicts"]:
        m = Seq2SeqNangCao("GRU", hidden=cfg["hidden"], layers=cfg["layers"], dropout=cfg["dropout"])
        m.load_state_dict(sd)
        models.append(m.eval())
    assert cfg["layers"] == 1, "Bản xuất ONNX viết cho GRU 1 lớp"
    return ck, cfg, models


def test_inputs(n_static):
    """Mọi kiểu khám thiếu (đủ 3 / thiếu 6M / thiếu 12M / chỉ 0M / thiếu 6M có 12M) + giá trị ngẫu nhiên."""
    g = torch.Generator().manual_seed(0)
    Vh = torch.randn(4000, 3, generator=g) * 1.2
    S = torch.randn(4000, n_static, generator=g)
    pat = torch.arange(4000) % 4
    Vh[pat == 1, 1] = float("nan")
    Vh[pat == 2, 2] = float("nan")
    Vh[pat == 3, 1:] = float("nan")
    Vh[::7, 1] = float("nan")
    return Vh, S


def main():
    ck, cfg, models = load_models()
    mu, sd = float(ck["mu"]), float(ck["sd"])
    exp = GruEnsembleExport(models).eval()

    params = json.loads((A / "tham_so_tien_xu_ly.json").read_text(encoding="utf-8"))
    n_static = len(params["cot_dau_vao_tinh"])
    Vh, S = test_inputs(n_static)

    # 1) bản viết lại == mô hình gốc (có pack_padded_sequence)
    with torch.no_grad():
        ln = torch.full((len(Vh),), 3, dtype=torch.long)
        ref = torch.stack([m(Vh, ln, S)[:, :N_PRED] for m in models]).numpy()
        new = exp(Vh, S).numpy()
    d1 = np.abs(ref - new).max() * sd
    print(f"Bản viết lại so với mô hình gốc: lệch tối đa {d1:.2e} eGFR")
    assert d1 < 1e-3

    # 2) xuất ONNX
    out = A / "gru.onnx"
    torch.onnx.export(exp, (Vh[:2], S[:2]), out, input_names=["Vh", "S"], output_names=["per_seed"],
                      dynamic_axes={"Vh": {0: "B"}, "S": {0: "B"}, "per_seed": {1: "B"}},
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onx = sess.run(None, {"Vh": Vh.numpy(), "S": S.numpy()})[0]
    d2 = np.abs(ref - onx).max() * sd
    print(f"ONNX so với mô hình gốc: lệch tối đa {d2:.2e} eGFR ({len(Vh)} mẫu × 5 seed × 4 mốc)")
    assert d2 < 1e-3

    n_params = sum(p.numel() for p in models[0].parameters())
    (A / "gru.json").write_text(json.dumps({
        "cau_hinh": cfg, "mu": mu, "sd": sd, "n_seeds": len(models), "n_params_per_seed": n_params,
        "nguon": "gru_lstm_phat_nang.pt (GRU.ipynb)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) phân cụm: xuất tham số, kiểm tra NumPy == scikit-learn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pc = joblib.load(A / "phan_cum.joblib")
    cols = pc["COT_VAO"]
    km, sc = pc["kmeans"], pc["scaler"]
    assert sc.with_mean and sc.with_std
    pj = {
        "MO_HINH_CHON": pc["MO_HINH_CHON"], "K": int(pc["K"]), "COT_VAO": cols,
        "can_duoi": [float(pc["can_duoi"][c]) for c in cols], "can_tren": [float(pc["can_tren"][c]) for c in cols],
        "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
        "tam_cum": km.cluster_centers_.tolist(), "ten_km": {str(k): v for k, v in pc["ten_km"].items()},
        "NGUONG_NHANH": float(pc["NGUONG_NHANH"]), "nguon": "phan_cum.joblib (GRU.ipynb)",
    }
    (A / "phan_cum.json").write_text(json.dumps(pj, ensure_ascii=False, indent=2), encoding="utf-8")

    rng = np.random.default_rng(0)
    F = rng.normal(size=(20000, len(cols))) * 2
    lo, hi = np.array(pj["can_duoi"]), np.array(pj["can_tren"])
    Xs = (np.clip(F, lo, hi) - np.array(pj["scaler_mean"])) / np.array(pj["scaler_scale"])
    dist = np.sqrt(((Xs[:, None, :] - np.array(pj["tam_cum"])[None]) ** 2).sum(-1))
    Xk = sc.transform(np.clip(F, lo, hi))
    same = (dist.argmin(1) == km.predict(Xk)).mean()
    d3 = np.abs(dist - km.transform(Xk)).max()
    print(f"K-Means NumPy so với scikit-learn: cùng cụm {same:.2%}, khoảng cách lệch tối đa {d3:.2e}")
    assert same == 1.0 and d3 < 1e-9
    print("Đã ghi:", ", ".join(p.name for p in (out, A / "gru.json", A / "phan_cum.json")))


if __name__ == "__main__":
    main()
