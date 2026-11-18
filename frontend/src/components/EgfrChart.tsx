import { CartesianGrid, ComposedChart, Legend, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { STAGE_BANDS } from '../lib/labels'

type Pt = { month: number; egfr: number | null }
interface Props {
  history: Pt[]
  predicted: Pt[]
  /** eGFR thật 18–36 tháng (chỉ có với bệnh nhân mẫu từ tập Test) */
  actual?: Pt[] | null
  height?: number
}

const COLORS = { actual: '#0f172a', pred: '#0f766e', bridge: '#94a3b8', truth: '#c2410c' }

export function EgfrChart({ history, predicted, actual, height = 340 }: Props) {
  const months = [0, 6, 12, 18, 24, 30, 36]
  const hist = new Map(history.filter((h) => h.egfr != null).map((h) => [h.month, h.egfr as number]))
  const pred = new Map(predicted.map((r) => [r.month, r.egfr]))
  const truth = new Map((actual ?? []).filter((a) => a.egfr != null).map((a) => [a.month, a.egfr]))
  const lastObsMonth = hist.size ? Math.max(...hist.keys()) : null

  const data = months.map((m) => ({
    month: m,
    measured: hist.get(m) ?? null,
    predicted: pred.get(m) ?? null,
    truth: truth.get(m) ?? null,
    // đường nối từ lần đo thật cuối cùng tới mốc dự đoán đầu tiên
    bridge: m === lastObsMonth ? hist.get(m)! : m === 18 ? pred.get(18) ?? null : null,
  }))
  const all = data.flatMap((d) => [d.measured, d.predicted, d.truth]).filter((v): v is number => v != null)
  const top = Math.max(...all, 0) > 85 ? 120 : 90
  const yTicks = [0, 15, 30, 45, 60, 90, ...(top > 90 ? [120] : [])]

  return (
    <div role="img" aria-label="Biểu đồ eGFR đã đo và eGFR dự đoán theo tháng" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 56, left: 0, bottom: 16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <ReferenceArea x1={12} x2={36} fill="#f0fdfa" fillOpacity={0.8} ifOverflow="visible" />
          <XAxis dataKey="month" type="number" domain={[0, 36]} ticks={months}
            label={{ value: 'Tháng (tính từ lần đo đầu)', position: 'insideBottom', offset: -8, fontSize: 12, fill: '#475569' }} tick={{ fontSize: 12 }} />
          <YAxis domain={[0, top]} ticks={yTicks} tick={{ fontSize: 12 }}
            label={{ value: 'eGFR (mL/phút/1,73m²)', angle: -90, position: 'insideLeft', fontSize: 12, fill: '#475569', dx: 4 }} />
          {STAGE_BANDS.slice(0, 5).filter((b) => b.min <= top).map((b) => (
            <ReferenceLine key={b.stage} y={b.min} stroke="#cbd5e1" strokeDasharray="4 4" ifOverflow="hidden"
              label={{ value: `${b.stage} ≥ ${b.min}`, position: 'right', fontSize: 11, fill: '#64748b' }} />
          ))}
          <Tooltip formatter={(v, name) => [typeof v === 'number' ? `${v.toFixed(1)}` : '—', name]} labelFormatter={(l) => `Tháng ${l}`} />
          <Legend verticalAlign="top" height={32} wrapperStyle={{ fontSize: 12 }} />
          <Line name="Nối lần đo cuối → dự đoán" dataKey="bridge" stroke={COLORS.bridge} strokeDasharray="2 4" dot={false} connectNulls legendType="plainline" isAnimationActive={false} />
          <Line name="eGFR đã đo (đầu vào)" dataKey="measured" stroke={COLORS.actual} strokeWidth={2.5} dot={{ r: 5, fill: COLORS.actual }} connectNulls isAnimationActive={false} />
          <Line name="eGFR dự đoán (GRU)" dataKey="predicted" stroke={COLORS.pred} strokeWidth={2.5} strokeDasharray="7 4" dot={{ r: 5, fill: '#fff', stroke: COLORS.pred, strokeWidth: 2 }} connectNulls isAnimationActive={false} />
          {actual && <Line name="eGFR thực tế (tập Test)" dataKey="truth" stroke={COLORS.truth} strokeWidth={2} dot={{ r: 4, fill: COLORS.truth }} connectNulls isAnimationActive={false} />}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
