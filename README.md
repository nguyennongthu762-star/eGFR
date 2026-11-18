# CKD eGFR – minh hoạ mô hình GRU + K-Means

Một trang web duy nhất, mở ra là dùng ngay (không đăng nhập, không lưu dữ liệu): nhập dữ liệu của **một bệnh nhân**,
bấm **Dự đoán**, hệ thống chạy lần lượt 5 bước bằng đúng các tệp mô hình đã huấn luyện:

```
1. Kiểm tra dữ liệu → 2. Tiền xử lý → 3. GRU dự đoán eGFR tháng 18–36 → 4. K-Means phân nhóm → 5. Kết quả & cảnh báo
```

Dữ liệu đầu vào: **chọn 1 trong 171 bệnh nhân tập Test** (lọc theo nhóm, tìm theo ID, chọn ngẫu nhiên; form tự điền giá trị gốc)
hoặc **nhập tay** 12 ô. Mỗi bước hiển thị dữ liệu trung gian thật (z-score, đầu ra từng seed GRU, đặc trưng và khoảng cách tới tâm cụm) và thời gian chạy.
Chỉ phục vụ nghiên cứu/minh hoạ; **không** chẩn đoán, không thay thế đánh giá của nhân viên y tế.

```
ckd-hm/
├─ backend/                 FastAPI + PyTorch (CPU) + scikit-learn
│  ├─ app/ml/network.py     kiến trúc Seq2Seq GRU (sao chép nguyên văn GRU.ipynb, ô 4)
│  ├─ app/ml/predictor.py   nạp mô hình + chạy 5 bước (run_pipeline)
│  ├─ app/routers/predict.py  GET /api/health · GET /api/test-patients · POST /api/predict
│  ├─ app/data/test_patients.json  171 bệnh nhân tập Test (giá trị gốc CKD-ROUTE + kết quả notebook + eGFR thật)
│  ├─ ml_assets/            gru_lstm_phat_nang.pt · phan_cum.joblib · tham_so_tien_xu_ly.json
│  ├─ scripts/              verify_model.py (kiểm chứng với notebook) · build_test_patients.py
│  └─ tests/                pytest (10 test, gồm kiểm tra 171 BN tập Test khớp notebook)
├─ frontend/                React + TypeScript + Vite + Tailwind + Recharts (1 trang)
├─ Demo_HM/                 (KHÔNG commit) notebook huấn luyện GRU.ipynb, train/val/test.xlsx, CSV kết quả
└─ render.yaml              triển khai API lên Render (không cần cơ sở dữ liệu)
```

## 1. Mô hình được dùng
- **GRU**: Seq2Seq GRU + Attention, tập hợp 5 seed, hidden 64, 1 lớp, dropout 0.3 (60.929 tham số/seed).
  Đầu vào: eGFR tháng 0/6/12 (tháng 6/12 có thể thiếu) + 13 biến tĩnh đã chuẩn hoá. Đầu ra: eGFR tháng 18/24/30/36.
  MAE Train/Validation/Test = **3,693 / 4,179 / 3,508** mL/phút/1,73m².
- **K-Means** (K = 3): 5 đặc trưng = độ dốc eGFR, thay đổi tổng tới tháng 36, UPCR_z, Alb_z, SBP_z → cắt phân vị 5–95% của Train → StandardScaler → nhóm *Chuyển biến xấu / Ổn định / Cải thiện*.
- 13 biến tĩnh: tuổi, giới (mã 1/2), tăng huyết áp, đái tháo đường, bệnh tim mạch, nguyên nhân CKD (mã 1–4, one-hot), UPCR (log1p, z-score; thiếu → giá trị điền của Train + cờ `UPCR_thieu`), albumin, huyết áp tâm thu.
  Tham số chuẩn hoá lấy từ tập Train (`tham_so_tien_xu_ly.json`).
- **Đã kiểm chứng**: `scripts/verify_model.py` chạy lại mô hình trên train/val/test và so với CSV của notebook: MAE khớp, sai khác dự đoán < 2·10⁻⁵, nhóm trùng 100%.
  Test `test_all_test_patients_reproduce_notebook` chạy cả 171 bệnh nhân tập Test qua API: eGFR và nhóm đều khớp notebook.

### Mã giới tính và nguyên nhân CKD
CKD-ROUTE lưu dạng số; ý nghĩa đã đối chiếu với Bảng 2 của bài báo gốc (Iimori et al., PLoS ONE 2018, doi:10.1371/journal.pone.0190493):
tỉ lệ nam 70,2% (n = 927) và 69,3% (n = 352); nguyên nhân 7,7% / 59,7% / 6,8% / 25,9% ở nhóm đạm niệu bình thường (n = 352)
khớp đúng với mã 1 / 2 / 3 / 4 trong dữ liệu.

| Biến | Mã | Ý nghĩa |
|---|---|---|
| gender | 1 / 2 | Nam / Nữ |
| etiology of CKD | 1 / 2 / 3 / 4 | Bệnh thận đái tháo đường / Xơ hoá thận (nephrosclerosis) / Viêm cầu thận / Khác |

## 2. Chạy trên máy
**Nhanh nhất (Windows):** nhấp đúp `chay.bat` ở thư mục gốc (hoặc gõ `.\chay.bat`). Script mở backend và frontend
trong 2 cửa sổ rồi mở http://localhost:5173. Đóng 2 cửa sổ đó để tắt. Cần cài đặt lần đầu như dưới đây.

Yêu cầu: Python 3.12, Node 20+.
```bash
# Backend (cửa sổ 1)
cd backend
python -m venv .venv && .venv/Scripts/activate        # Linux/mac: source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --port 8000                       # Swagger: http://localhost:8000/api/docs
pytest -q

# Frontend (cửa sổ 2)
cd frontend && npm install && npm run dev               # mở http://localhost:5173 (proxy /api -> :8000)
```
Kiểm chứng với notebook (cần thư mục `Demo_HM`): `python scripts/verify_model.py ../Demo_HM`.

## 3. API
| Method | Đường dẫn | Mô tả |
|---|---|---|
| GET | `/api/health` | `model_ready`, cấu hình mô hình đã nạp |
| GET | `/api/test-patients` | 171 bệnh nhân tập Test (giá trị gốc) + dự đoán của notebook + eGFR thật |
| POST | `/api/predict` | chạy 5 bước, trả về `steps[]` (mỗi bước: `key`, `title`, `ms`, `detail`) và `total_ms` |

```http
POST /api/predict
{"egfr_0m":45.5,"egfr_6m":43.0,"egfr_12m":45.3,"age":65,"gender":1,"etiology":2,
 "hypertension":true,"diabetes":false,"cvd":false,"sbp":140,"albumin":3.8,"upcr":0.9}
```
Dữ liệu sai kiểu/ngoài khoảng → `422 {"error":"validation_error","details":[{"field","label","message"}]}`.
Mô hình chưa nạp được → `503 model_not_ready` (không bao giờ trả kết quả giả).

## 4. Triển khai
- **Backend (Render)**: tạo từ `render.yaml` (Docker, root `backend`). Biến môi trường: `ENVIRONMENT=production`, `CORS_ORIGINS=https://<app>.vercel.app`. PyTorch CPU cần vài trăm MB RAM.
- **Frontend (Vercel)**: Root Directory `frontend`, Build `npm run build`, Output `dist`, biến `VITE_API_URL=https://<api>.onrender.com`.
- Kiểm tra: `GET <api>/api/health` → `{"status":"ok","model_ready":true}`.

## 5. Hạn chế
- Không có khoảng tin cậy; “dao động giữa 5 seed” chỉ thể hiện mức đồng thuận giữa các mô hình con.
- Nhãn nhóm là tương đối so với dữ liệu huấn luyện (nhóm “Cải thiện” không nhất thiết có eGFR tăng).
