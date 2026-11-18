export type Cluster = 'Chuyển biến xấu' | 'Ổn định' | 'Cải thiện'

export interface PatientInput {
  egfr_0m: number
  egfr_6m: number | null
  egfr_12m: number | null
  age: number
  gender: number
  etiology: number
  hypertension: boolean
  diabetes: boolean
  cvd: boolean
  sbp: number
  albumin: number
  upcr: number | null
}

export interface Health {
  model_ready: boolean
  model_error: string | null
  model: {
    gru_config?: Record<string, number>
    n_gru_seeds?: number
    n_params_per_seed?: number
    cluster_model?: string
    n_clusters?: number
    fast_decline_threshold?: number
  }
}

export interface TestPatient {
  id: number
  input: PatientInput
  /** trường bị thiếu trong dữ liệu gốc, đã điền bằng giá trị điền thiếu của tập Train */
  filled: string[]
  notebook: { cluster: Cluster; predicted: Record<string, number> }
  actual: Record<string, number | null>
}

export interface ValidateDetail { checks: { status: 'ok' | 'warn'; text: string }[]; n_visits: number }
export interface PreprocessDetail {
  static: { name: string; label: string; raw: string | number | null; transform: string; value: number }[]
  sequence: { month: number; egfr: number | null; observed: boolean; z: number | null }[]
  egfr_zscore: string
  note: string
}
export interface GruDetail {
  predicted: { month: number; egfr: number; stage: string; seed_min: number; seed_max: number }[]
  history: { month: number; egfr: number | null }[]
  last_observed: number
  n_seeds: number
  config: Record<string, number>
}
export interface KmeansDetail {
  features: { name: string; label: string; value: number; clipped: number; scaled: number }[]
  distances: { cluster: Cluster; distance: number }[]
  cluster: Cluster
  risk_score: number
  slope_per_year: number
  total_change: number
}
export interface ConcludeDetail {
  cluster: Cluster
  stage_36m: string
  egfr_36m: number
  last_observed: number
  change_to_36m: number
  slope_per_year: number
  confidence: string
  warnings: { level: 'high' | 'info'; text: string }[]
}

export type Step =
  | { key: 'validate'; title: string; ms: number; detail: ValidateDetail }
  | { key: 'preprocess'; title: string; ms: number; detail: PreprocessDetail }
  | { key: 'gru'; title: string; ms: number; detail: GruDetail }
  | { key: 'kmeans'; title: string; ms: number; detail: KmeansDetail }
  | { key: 'conclude'; title: string; ms: number; detail: ConcludeDetail }

export interface PredictResponse { input: PatientInput; steps: Step[]; total_ms: number }
