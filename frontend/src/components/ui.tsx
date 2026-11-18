import clsx from 'clsx'
import { AlertTriangle, CheckCircle2, Info, Loader2, MinusCircle, TrendingDown, TrendingUp } from 'lucide-react'
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from 'react'
import type { Cluster } from '../lib/types'

// ---------- Button ----------
type BtnProps = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'secondary' | 'ghost'; loading?: boolean }
export function Button({ variant = 'primary', loading, className, children, disabled, ...p }: BtnProps) {
  return (
    <button
      {...p}
      disabled={disabled || loading}
      className={clsx(
        'inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 min-h-10',
        variant === 'primary' && 'bg-brand-700 text-white hover:bg-brand-800',
        variant === 'secondary' && 'border border-slate-300 bg-white text-slate-700 hover:bg-slate-50',
        variant === 'ghost' && 'text-slate-600 hover:bg-slate-100',
        className,
      )}
    >
      {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
      {children}
    </button>
  )
}

// ---------- Card ----------
export function Card({ title, actions, children, className }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={clsx('rounded-xl border border-slate-200 bg-white', className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-5 py-3">
          <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
          {actions}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  )
}

// ---------- Alert ----------
export function Alert({ kind = 'info', title, children }: { kind?: 'info' | 'warn' | 'error' | 'success'; title?: string; children: ReactNode }) {
  const Icon = { info: Info, warn: AlertTriangle, error: AlertTriangle, success: CheckCircle2 }[kind]
  const cls = {
    info: 'border-sky-200 bg-sky-50 text-sky-900',
    warn: 'border-amber-300 bg-amber-50 text-amber-900',
    error: 'border-red-300 bg-red-50 text-red-900',
    success: 'border-emerald-300 bg-emerald-50 text-emerald-900',
  }[kind]
  return (
    <div role={kind === 'error' ? 'alert' : 'status'} className={clsx('flex gap-3 rounded-lg border p-3 text-sm', cls)}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <div>
        {title && <p className="font-semibold">{title}</p>}
        <div>{children}</div>
      </div>
    </div>
  )
}

export function Disclaimer() {
  return (
    <p className="text-xs text-slate-500">
      Kết quả là ước lượng của mô hình học máy huấn luyện trên bộ dữ liệu CKD-ROUTE, chỉ phục vụ nghiên cứu và minh hoạ.
      Không phải chẩn đoán và không thay thế đánh giá của nhân viên y tế.
    </p>
  )
}

// ---------- Badge nhóm ----------
const CLUSTER_STYLE: Record<Cluster, { cls: string; Icon: typeof TrendingDown }> = {
  'Chuyển biến xấu': { cls: 'border-red-300 bg-red-50 text-red-800', Icon: TrendingDown },
  'Ổn định': { cls: 'border-amber-300 bg-amber-50 text-amber-900', Icon: MinusCircle },
  'Cải thiện': { cls: 'border-emerald-300 bg-emerald-50 text-emerald-800', Icon: TrendingUp },
}
export function ClusterBadge({ cluster, size = 'md' }: { cluster: Cluster | string; size?: 'md' | 'lg' }) {
  const s = CLUSTER_STYLE[cluster as Cluster]
  if (!s) return <span>{cluster}</span>
  return (
    <span className={clsx('inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border font-medium', s.cls, size === 'lg' ? 'px-3.5 py-1.5 text-base' : 'px-2.5 py-0.5 text-xs')}>
      <s.Icon className={size === 'lg' ? 'h-5 w-5' : 'h-3.5 w-3.5'} aria-hidden />
      {cluster}
    </span>
  )
}

// ---------- Form ----------
export function Field({ label, hint, unit, error, children, required }: { label: string; hint?: string; unit?: string; error?: string; children: ReactNode; required?: boolean }) {
  return (
    <label className="block text-sm">
      <span className="mb-1 block font-medium text-slate-700">
        {label} {required && <span className="text-red-600" aria-label="bắt buộc">*</span>}
        {unit && <span className="ml-1 font-normal text-slate-500">({unit})</span>}
      </span>
      {children}
      {hint && !error && <span className="mt-1 block text-xs text-slate-500">{hint}</span>}
      {error && <span role="alert" className="mt-1 block text-xs text-red-700">{error}</span>}
    </label>
  )
}
const inputCls = 'block w-full rounded-lg border bg-white px-3 py-2 text-sm min-h-10 focus:border-brand-600 disabled:bg-slate-100'
export const Input = ({ invalid, ...p }: InputHTMLAttributes<HTMLInputElement> & { invalid?: boolean }) =>
  <input {...p} aria-invalid={invalid || undefined} className={clsx(inputCls, invalid ? 'border-red-400' : 'border-slate-300', p.className)} />
export const Select = (p: SelectHTMLAttributes<HTMLSelectElement>) => <select {...p} className={clsx(inputCls, 'border-slate-300', p.className)} />

export function Check({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex min-h-10 cursor-pointer items-center gap-2 text-sm text-slate-700">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="h-4 w-4 rounded border-slate-300 accent-brand-700" />
      {label}
    </label>
  )
}

// ---------- Bảng ----------
export function Table({ head, children }: { head: ReactNode[]; children: ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
            {head.map((h, i) => <th key={i} scope="col" className="whitespace-nowrap px-3 py-2 font-medium">{h}</th>)}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">{children}</tbody>
      </table>
    </div>
  )
}
export const Td = ({ children, className }: { children: ReactNode; className?: string }) => <td className={clsx('px-3 py-2 align-middle', className)}>{children}</td>
