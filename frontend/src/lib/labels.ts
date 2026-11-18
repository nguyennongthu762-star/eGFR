import type { Cluster } from './types'

// Mã gốc của CKD-ROUTE (giới tính 1/2, nguyên nhân 1–4). Ý nghĩa đã đối chiếu với Bảng 2 của bài báo gốc
// (Iimori et al., PLoS ONE 2018): tỉ lệ nam 70,2%; nguyên nhân 7,7% / 59,7% / 6,8% / 25,9% ở nhóm n = 352
// khớp đúng với mã 1 / 2 / 3 / 4. Mô hình chỉ nhận con số, nhãn chỉ để hiển thị.
export const GENDER_LABEL: Record<number, string> = { 1: 'Nam', 2: 'Nữ' }
export const ETIOLOGY_LABEL: Record<number, string> = {
  1: 'Bệnh thận đái tháo đường',
  2: 'Xơ hoá thận (nephrosclerosis)',
  3: 'Viêm cầu thận',
  4: 'Nguyên nhân khác',
}
export const CLUSTERS: Cluster[] = ['Chuyển biến xấu', 'Ổn định', 'Cải thiện']
export const FIELD_NAME: Record<string, string> = { sbp: 'huyết áp tâm thu', albumin: 'albumin' }

export const STAGE_BANDS = [
  { stage: 'G1', min: 90, label: '≥ 90' },
  { stage: 'G2', min: 60, label: '60–89' },
  { stage: 'G3a', min: 45, label: '45–59' },
  { stage: 'G3b', min: 30, label: '30–44' },
  { stage: 'G4', min: 15, label: '15–29' },
  { stage: 'G5', min: 0, label: '< 15' },
]

export const CLUSTER_TEXT: Record<string, string> = {
  'Chuyển biến xấu': 'eGFR dự kiến giảm nhanh hơn phần lớn bệnh nhân trong dữ liệu huấn luyện.',
  'Ổn định': 'eGFR dự kiến giảm ở mức trung bình so với dữ liệu huấn luyện.',
  'Cải thiện': 'eGFR dự kiến giảm chậm hoặc giữ ổn định hơn so với dữ liệu huấn luyện (nhãn tương đối, không nhất thiết là eGFR tăng).',
}

export const num = (v: number | null | undefined, d = 1) => (v == null ? '—' : v.toFixed(d))
export const signed = (v: number | null | undefined, d = 1) => (v == null ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(d)}`)

/** ID bệnh nhân hiển thị đủ 6 chữ số như trong file Excel (vd 10412 -> 010412) */
export const fmtId = (id: number) => String(id).padStart(6, '0')
