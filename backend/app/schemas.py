"""Dữ liệu đầu vào của một bệnh nhân. Bước 1 'Kiểm tra dữ liệu' của quy trình: sai kiểu / ngoài khoảng -> 422."""
from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    # eGFR (mL/phút/1,73m²) tại 3 lần khám; tháng 0 bắt buộc, tháng 6/12 có thể thiếu
    egfr_0m: float = Field(ge=1, le=200)
    egfr_6m: float | None = Field(default=None, ge=1, le=200)
    egfr_12m: float | None = Field(default=None, ge=1, le=200)
    # thông tin nền
    age: float = Field(ge=18, le=110, description="tuổi (năm)")
    gender: int = Field(ge=1, le=2, description="1 = Nam, 2 = Nữ (mã gốc CKD-ROUTE)")
    etiology: int = Field(ge=1, le=4, description="1 = Bệnh thận đái tháo đường, 2 = Xơ hoá thận, "
                                                  "3 = Viêm cầu thận, 4 = Nguyên nhân khác (mã gốc CKD-ROUTE)")
    hypertension: bool
    diabetes: bool
    cvd: bool = Field(description="bệnh tim mạch")
    # chỉ số lâm sàng
    sbp: float = Field(ge=50, le=300, description="huyết áp tâm thu, mmHg")
    albumin: float = Field(ge=1.0, le=6.0, description="albumin huyết thanh, g/dL")
    upcr: float | None = Field(default=None, ge=0, le=50, description="UPCR, g/g; để trống nếu chưa đo")
