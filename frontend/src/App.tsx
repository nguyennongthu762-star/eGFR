import { Activity, AlertTriangle, Info, ServerCrash } from 'lucide-react'
import { useEffect, useState } from 'react'
import { EgfrChart } from './components/EgfrChart'
import { PatientForm } from './components/PatientForm'
import { EMPTY_FORM, fromInput, toInput, validate, type FormState } from './lib/form'
import { Alert, Card, ClusterBadge, Disclaimer, Table, Td } from './components/ui'
import { api, ApiError } from './lib/api'
import { CLUSTER_TEXT, STAGE_BANDS, fmtId, num, signed } from './lib/labels'
import type { ConcludeDetail, GruDetail, Health, PredictResponse, TestPatient } from './lib/types'

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [healthErr, setHealthErr] = useState<string | null>(null)
  const [patients, setPatients] = useState<TestPatient[]>([])
  const [form, setForm] = useState<FormState>(EMPTY_FORM)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [picked, setPicked] = useState<TestPatient | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [fail, setFail] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // Backend cần vài chục giây để nạp PyTorch + mô hình: thử lại mỗi 2 giây cho tới khi kết nối được
  useEffect(() => {
    let stop = false
    let retry: number | undefined
    const connect = async () => {
      try {
        const h = await api<Health>('/api/health')
        const pts = await api<TestPatient[]>('/api/test-patients')
        if (stop) return
        setHealth(h); setPatients(pts); setHealthErr(null)
      } catch (e) {
        if (stop) return
        setHealthErr(e instanceof Error ? e.message : 'Không kết nối được backend')
        retry = window.setTimeout(connect, 2000)
      }
    }
    connect()
    return () => { stop = true; clearTimeout(retry) }
  }, [])

  const clearRun = () => {
    setResult(null); setFail(null)
  }
  const onChange = (f: FormState) => { setForm(f); setPicked(null); setErrors({}) }
  const onPick = (p: TestPatient) => { clearRun(); setForm(fromInput(p.input)); setPicked(p); setErrors({}) }
  const onReset = () => { clearRun(); setForm(EMPTY_FORM); setPicked(null); setErrors({}) }

  const run = async () => {
    clearRun()
    // Bước 1 phía giao diện: chặn sớm lỗi nhập để người dùng sửa ngay
    const errs = validate(form)
    setErrors(errs)
    if (Object.keys(errs).length) {
      setFail('Dữ liệu chưa hợp lệ: hãy sửa các ô được đánh dấu đỏ rồi bấm “Dự đoán” lại.')
      return
    }
    setBusy(true)
    try {
      setResult(await api<PredictResponse>('/api/predict', toInput(form)))
    } catch (ex) {
      if (ex instanceof ApiError && ex.status === 422) {
        const fe: Record<string, string> = {}
        ex.details?.forEach((d) => { fe[d.field] = `${d.label ?? d.field}: ${d.message}` })
        setErrors(fe)
        setFail(ex.details?.length ? 'Máy chủ từ chối dữ liệu: ' + ex.details.map((d) => `${d.label ?? d.field} – ${d.message}`).join('; ') : ex.message)
      } else {
        setFail(ex instanceof Error ? ex.message : 'Có lỗi xảy ra')
      }
    } finally {
      setBusy(false)
    }
  }

  const modelReady = health?.model_ready === true
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-4 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-lg bg-brand-700 text-white"><Activity className="h-5 w-5" aria-hidden /></div>
            <div>
              <h1 className="text-lg font-semibold text-slate-900 sm:text-xl">Dự đoán tiến triển bệnh thận mạn qua eGFR</h1>
              <p className="text-sm text-slate-600">GRU dự đoán eGFR tháng 18–36 · K-Means phân nhóm tiến triển</p>
            </div>
          </div>
          <ModelStatus health={health} error={healthErr} />
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
        <div className="lg:sticky lg:top-6 lg:self-start">
          <PatientForm form={form} errors={errors} onChange={onChange} onSubmit={run} onReset={onReset}
            patients={patients} picked={picked} onPick={onPick} busy={busy} disabled={!modelReady} />
        </div>

        <div className="min-w-0 space-y-5">
          {fail && <Alert kind="error">{fail}</Alert>}
          {!result && !fail && <Intro />}
          {result && <Results r={result} example={picked} />}
          <Disclaimer />
        </div>
      </main>
    </div>
  )
}

function ModelStatus({ health, error }: { health: Health | null; error: string | null }) {
  if (error) return <span className="inline-flex items-center gap-1.5 rounded-full border border-red-300 bg-red-50 px-3 py-1 text-xs font-medium text-red-800"><ServerCrash className="h-3.5 w-3.5" aria-hidden /> Chưa kết nối được backend · đang thử lại…</span>
  if (!health) return <span className="text-xs text-slate-500">Đang kết nối máy chủ… (lần đầu có thể mất khoảng 1 phút)</span>
  if (!health.model_ready) return <span title={health.model_error ?? ''} className="inline-flex items-center gap-1.5 rounded-full border border-red-300 bg-red-50 px-3 py-1 text-xs font-medium text-red-800"><AlertTriangle className="h-3.5 w-3.5" aria-hidden /> Chưa nạp được mô hình</span>
  return null
}

function Intro() {
  return (
    <Card>
      <div className="flex gap-3 text-sm text-slate-700">
        <Info className="mt-0.5 h-5 w-5 shrink-0 text-brand-700" aria-hidden />
        <div className="space-y-2">
          <p>Chọn một bệnh nhân từ tập Test hoặc nhập tay dữ liệu của một bệnh nhân, rồi bấm <b>Dự đoán</b>. Hệ thống chạy lần lượt 5 bước sau bằng đúng các tệp mô hình đã huấn luyện:</p>
          <ol className="list-decimal space-y-1 pl-5">
            <li><b>Kiểm tra dữ liệu</b>: trường bắt buộc, khoảng giá trị, lần khám thiếu.</li>
            <li><b>Tiền xử lý</b>: điền thiếu, log, z-score bằng tham số tính trên tập Train.</li>
            <li><b>GRU</b> dự đoán eGFR tại tháng 18, 24, 30, 36 từ eGFR tháng 0, 6, 12 và 13 biến lâm sàng.</li>
            <li><b>K-Means</b> xếp bệnh nhân vào nhóm <i>Chuyển biến xấu / Ổn định / Cải thiện</i> từ xu hướng eGFR, UPCR, albumin, huyết áp.</li>
            <li><b>Kết quả và cảnh báo</b>: biểu đồ, giai đoạn eGFR, nhóm tiến triển, cảnh báo.</li>
          </ol>
        </div>
      </div>
    </Card>
  )
}

function Results({ r, example }: { r: PredictResponse; example: TestPatient | null }) {
  const c = r.steps.find((s) => s.key === 'conclude')!.detail as ConcludeDetail
  const g = r.steps.find((s) => s.key === 'gru')!.detail as GruDetail
  const actual = example ? Object.entries(example.actual).map(([m, v]) => ({ month: Number(m), egfr: v })) : null
  return (
    <>
      <Card title="Kết quả">
        <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
          <div><p className="text-xs text-slate-500">Nhóm tiến triển (K-Means)</p><div className="mt-1.5"><ClusterBadge cluster={c.cluster} size="lg" /></div></div>
          <div><p className="text-xs text-slate-500">eGFR dự đoán tháng 36</p><p className="text-2xl font-semibold">{num(c.egfr_36m)} <span className="text-base font-medium text-slate-600">{c.stage_36m}</span></p></div>
          <div><p className="text-xs text-slate-500">Thay đổi so với lần đo cuối ({num(c.last_observed)})</p><p className="text-2xl font-semibold">{signed(c.change_to_36m)}</p></div>
          <div><p className="text-xs text-slate-500">Tốc độ thay đổi eGFR</p><p className="text-2xl font-semibold">{signed(c.slope_per_year)} <span className="text-sm font-normal text-slate-500">/năm</span></p></div>
        </div>
        <p className="mt-4 text-sm text-slate-700">{CLUSTER_TEXT[c.cluster]} Độ tin cậy dữ liệu đầu vào: <b>{c.confidence}</b>.</p>
        <div className="mt-4 space-y-2">
          {c.warnings.length === 0 && <Alert kind="success">Không có cảnh báo.</Alert>}
          {c.warnings.map((w) => <Alert key={w.text} kind={w.level === 'high' ? 'warn' : 'info'}>{w.text}</Alert>)}
        </div>
      </Card>

      <Card title="eGFR đã đo và dự đoán">
        <EgfrChart history={g.history} predicted={g.predicted} actual={actual} />
        <Table head={['Tháng', 'eGFR dự đoán', 'Giai đoạn', ...(example ? ['Notebook', 'eGFR thực tế'] : [])]}>
          {g.predicted.map((p) => (
            <tr key={p.month}>
              <Td>{p.month}</Td><Td className="font-medium">{num(p.egfr, 2)}</Td>
              <Td>{p.stage} <span className="text-xs text-slate-500">({STAGE_BANDS.find((b) => b.stage === p.stage)?.label})</span></Td>
              {example && <><Td>{num(example.notebook.predicted[p.month], 2)}</Td><Td>{num(example.actual[p.month], 2)}</Td></>}
            </tr>
          ))}
        </Table>
        {example && (
          <p className="mt-2 text-xs text-slate-500">
            BN {fmtId(example.id)} thuộc tập Test. Cột “Notebook” là kết quả GRU.ipynb lưu lại (nhóm: {example.notebook.cluster}) để đối chiếu hệ thống chạy đúng mô hình;
            cột “eGFR thực tế” là giá trị đo thực tế ở tháng 18–36, mô hình không được thấy khi dự đoán.
          </p>
        )}
      </Card>
    </>
  )
}
