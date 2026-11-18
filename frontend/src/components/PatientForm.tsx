import { Database, Eraser, Play, Shuffle } from 'lucide-react'
import { useMemo, useState, type FormEvent } from 'react'
import type { FormState } from '../lib/form'
import { CLUSTERS, ETIOLOGY_LABEL, FIELD_NAME, GENDER_LABEL, fmtId, num } from '../lib/labels'
import type { TestPatient } from '../lib/types'
import { Button, Card, Check, Field, Input, Select } from './ui'

interface Props {
  form: FormState
  errors: Record<string, string>
  onChange: (f: FormState) => void
  onSubmit: () => void
  onReset: () => void
  patients: TestPatient[]
  picked: TestPatient | null
  onPick: (p: TestPatient) => void
  busy: boolean
  disabled: boolean
}

export function PatientForm({ form, errors, onChange, onSubmit, onReset, patients, picked, onPick, busy, disabled }: Props) {
  const set = <K extends keyof FormState>(k: K, v: FormState[K]) => onChange({ ...form, [k]: v })
  const numField = (k: keyof FormState, label: string, opts: { placeholder?: string; required?: boolean } = {}) => (
    <Field label={label} required={opts.required} error={errors[k]}>
      <Input inputMode="decimal" placeholder={opts.placeholder} value={form[k] as string} invalid={!!errors[k]} onChange={(e) => set(k, e.target.value)} />
    </Field>
  )
  const submit = (e: FormEvent) => { e.preventDefault(); onSubmit() }

  return (
    <Card title="Dữ liệu bệnh nhân" actions={<Button type="button" variant="ghost" className="min-h-8 px-2 py-1" onClick={onReset}><Eraser className="h-4 w-4" aria-hidden /> Xoá</Button>}>
      {patients.length > 0 && <TestPatientPicker patients={patients} picked={picked} onPick={onPick} />}

      <form onSubmit={submit} noValidate className="space-y-5">
        <fieldset>
          <legend className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">eGFR đã đo (mL/phút/1,73m²)</legend>
          <div className="grid grid-cols-3 gap-3">
            {numField('egfr_0m', 'Tháng 0', { required: true })}
            {numField('egfr_6m', 'Tháng 6')}
            {numField('egfr_12m', 'Tháng 12')}
          </div>
          <p className="mt-1 text-xs text-slate-500">Để trống tháng 6/12 nếu không đo; mô hình vẫn dự đoán được nhưng độ tin cậy thấp hơn.</p>
        </fieldset>

        <fieldset>
          <legend className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Thông tin chung</legend>
          <div className="grid grid-cols-2 gap-3">
            {numField('age', 'Tuổi', { required: true })}
            <Field label="Giới tính"><Select value={form.gender} onChange={(e) => set('gender', e.target.value)}>
              {Object.entries(GENDER_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</Select></Field>
            <div className="col-span-2">
              <Field label="Nguyên nhân bệnh thận mạn"><Select value={form.etiology} onChange={(e) => set('etiology', e.target.value)}>
                {Object.entries(ETIOLOGY_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</Select></Field>
            </div>
          </div>
          <div className="mt-2 flex flex-wrap gap-x-5">
            <Check label="Tăng huyết áp" checked={form.hypertension} onChange={(v) => set('hypertension', v)} />
            <Check label="Đái tháo đường" checked={form.diabetes} onChange={(v) => set('diabetes', v)} />
            <Check label="Bệnh tim mạch" checked={form.cvd} onChange={(v) => set('cvd', v)} />
          </div>
        </fieldset>

        <fieldset>
          <legend className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Xét nghiệm, huyết áp</legend>
          <div className="grid grid-cols-3 gap-3">
            {numField('sbp', 'HA tâm thu', { placeholder: 'mmHg', required: true })}
            {numField('albumin', 'Albumin', { placeholder: 'g/dL', required: true })}
            {numField('upcr', 'UPCR', { placeholder: 'g/g' })}
          </div>
          <p className="mt-1 text-xs text-slate-500">Đơn vị: HA tâm thu mmHg · Albumin g/dL · UPCR g/g. Để trống UPCR nếu chưa đo.</p>
        </fieldset>

        <Button type="submit" className="w-full" loading={busy} disabled={disabled}>
          {!busy && <Play className="h-4 w-4" aria-hidden />} {busy ? 'Đang chạy…' : 'Dự đoán'}
        </Button>
      </form>
    </Card>
  )
}

/** Cách A: chọn 1 trong 171 bệnh nhân tập Test để điền sẵn form (vẫn sửa tay được = cách C). */
function TestPatientPicker({ patients, picked, onPick }: { patients: TestPatient[]; picked: TestPatient | null; onPick: (p: TestPatient) => void }) {
  const [group, setGroup] = useState('')
  const [query, setQuery] = useState('')
  // ID trong Excel có số 0 ở đầu (010412) còn dữ liệu lưu dạng số (10412): so khớp theo dạng đủ 6 chữ số
  const list = useMemo(() => {
    const q = query.trim()
    return patients.filter((p) => (!group || p.notebook.cluster === group) && (!q || fmtId(p.id).includes(q)))
  }, [patients, group, query])
  const onQuery = (q: string) => {
    setQuery(q)
    const id = Number(q.trim())
    const exact = q.trim().length >= 5 && patients.find((p) => p.id === id)
    if (exact) { setGroup(''); onPick(exact) }   // gõ đúng đủ 1 ID -> chọn luôn
  }
  const label = (p: TestPatient) => {
    const e = p.input
    const v = [e.egfr_0m, e.egfr_6m, e.egfr_12m].map((x) => (x == null ? '—' : num(x, 1))).join(' / ')
    return `BN ${fmtId(p.id)} · eGFR 0/6/12 tháng: ${v}`
  }
  const random = () => { if (list.length) onPick(list[Math.floor(Math.random() * list.length)]) }

  return (
    <div className="mb-5 rounded-lg border border-slate-200 bg-slate-50 p-3">
      <p className="mb-2 flex items-center gap-1.5 text-sm font-medium text-slate-800">
        <Database className="h-4 w-4 text-brand-700" aria-hidden /> Chọn bệnh nhân từ tập Test ({patients.length} BN)
      </p>
      <div className="grid grid-cols-[minmax(0,1fr)_5.5rem_auto] gap-2">
        <Select aria-label="Lọc theo nhóm notebook" value={group} onChange={(e) => setGroup(e.target.value)}>
          <option value="">Tất cả nhóm</option>
          {CLUSTERS.map((c) => <option key={c} value={c}>{c} ({patients.filter((p) => p.notebook.cluster === c).length})</option>)}
        </Select>
        <Input aria-label="Tìm theo ID" inputMode="numeric" placeholder="Tìm ID" value={query} onChange={(e) => onQuery(e.target.value)} />
        <Button type="button" variant="secondary" className="px-3" onClick={random} disabled={!list.length} title="Chọn ngẫu nhiên" aria-label="Chọn ngẫu nhiên">
          <Shuffle className="h-4 w-4" aria-hidden />
        </Button>
      </div>
      <Select aria-label="Bệnh nhân" className="mt-2" value={picked && list.some((p) => p.id === picked.id) ? picked.id : ''}
        onChange={(e) => { const p = patients.find((x) => x.id === Number(e.target.value)); if (p) onPick(p) }}>
        <option value="" disabled>{list.length ? `— Chọn 1 trong ${list.length} bệnh nhân —` : 'Không có bệnh nhân phù hợp'}</option>
        {list.map((p) => <option key={p.id} value={p.id}>{label(p)}</option>)}
      </Select>
      <p className="mt-2 text-xs text-slate-500">
        {picked
          ? <>Đã điền dữ liệu gốc của <b>BN {fmtId(picked.id)}</b>. Sửa bất kỳ ô nào thì chuyển sang nhập tay.</>
          : 'Tập Test không tham gia huấn luyện. Chọn 1 bộ để test, hoặc nhập tay trực tiếp các ô bên dưới.'}
        {picked && picked.filled.length > 0 && (
          <span className="mt-1 block text-amber-800">
            Dữ liệu gốc thiếu {picked.filled.map((f) => FIELD_NAME[f] ?? f).join(', ')}: đã điền giá trị điền thiếu của tập Train (như lúc huấn luyện).
          </span>
        )}
      </p>
    </div>
  )
}
