import type { PatientInput } from './types'

export interface FormState {
  egfr_0m: string; egfr_6m: string; egfr_12m: string
  age: string; gender: string; etiology: string
  hypertension: boolean; diabetes: boolean; cvd: boolean
  sbp: string; albumin: string; upcr: string
}
export const EMPTY_FORM: FormState = {
  egfr_0m: '', egfr_6m: '', egfr_12m: '', age: '', gender: '1', etiology: '1',
  hypertension: false, diabetes: false, cvd: false, sbp: '', albumin: '', upcr: '',
}

// Khoảng hợp lệ — giống hệt kiểm tra phía máy chủ (backend/app/schemas.py)
const RANGES: Record<string, [number, number, string, boolean]> = {
  egfr_0m: [1, 200, 'eGFR tháng 0', true], egfr_6m: [1, 200, 'eGFR tháng 6', false], egfr_12m: [1, 200, 'eGFR tháng 12', false],
  age: [18, 110, 'Tuổi', true], sbp: [50, 300, 'Huyết áp tâm thu', true], albumin: [1, 6, 'Albumin', true], upcr: [0, 50, 'UPCR', false],
}

export function fromInput(p: PatientInput): FormState {
  const s = (v: number | null) => (v == null ? '' : String(v))
  return {
    egfr_0m: s(p.egfr_0m), egfr_6m: s(p.egfr_6m), egfr_12m: s(p.egfr_12m), age: s(p.age), gender: String(p.gender),
    etiology: String(p.etiology), hypertension: p.hypertension, diabetes: p.diabetes, cvd: p.cvd,
    sbp: s(p.sbp), albumin: s(p.albumin), upcr: s(p.upcr),
  }
}

/** Kiểm tra phía giao diện. Trả về lỗi theo trường; rỗng = hợp lệ. */
export function validate(f: FormState): Record<string, string> {
  const e: Record<string, string> = {}
  for (const [k, [lo, hi, name, required]] of Object.entries(RANGES)) {
    const v = (f[k as keyof FormState] as string).trim().replace(',', '.')
    if (v === '') { if (required) e[k] = `${name} là bắt buộc`; continue }
    const n = Number(v)
    if (!Number.isFinite(n)) e[k] = `${name} phải là số`
    else if (n < lo || n > hi) e[k] = `${name} phải trong khoảng ${lo}–${hi}`
  }
  return e
}

export function toInput(f: FormState): PatientInput {
  const n = (v: string) => (v.trim() === '' ? null : Number(v.trim().replace(',', '.')))
  return {
    egfr_0m: n(f.egfr_0m)!, egfr_6m: n(f.egfr_6m), egfr_12m: n(f.egfr_12m), age: n(f.age)!,
    gender: Number(f.gender), etiology: Number(f.etiology), hypertension: f.hypertension, diabetes: f.diabetes, cvd: f.cvd,
    sbp: n(f.sbp)!, albumin: n(f.albumin)!, upcr: n(f.upcr),
  }
}
