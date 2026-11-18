const BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') ?? ''

export interface FieldError { field: string; label?: string; message: string }

export class ApiError extends Error {
  status: number
  code: string
  details?: FieldError[]
  constructor(status: number, code: string, message: string, details?: FieldError[]) {
    super(message)
    this.status = status
    this.code = code
    this.details = details
  }
}

export async function api<T>(path: string, body?: unknown): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, {
      method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch {
    throw new ApiError(0, 'network', 'Không kết nối được tới máy chủ (backend). Hãy kiểm tra backend đã chạy chưa.')
  }
  const data = await res.json().catch(() => null)
  if (!res.ok) {
    const d = data as { error?: string; message?: string; details?: FieldError[] } | null
    throw new ApiError(res.status, d?.error ?? 'error', d?.message ?? `Lỗi ${res.status}`, d?.details)
  }
  return data as T
}
