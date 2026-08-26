import {
  ArrowLeft,
  CheckCircle2,
  Download,
  FileCode2,
  ShieldAlert,
  WandSparkles,
  XCircle
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api'
import StatusBadge from '../components/StatusBadge'
import type {
  AnalysisDetail,
  FindingBundle,
  PatchBatch
} from '../types'

export default function AnalysisDetailPage() {
  const { id } = useParams()
  const [detail, setDetail] = useState<AnalysisDetail | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [error, setError] = useState('')
  const [patchBusy, setPatchBusy] = useState(false)

  const load = useCallback(async () => {
    if (!id) return
    try {
      const result = await api.get<AnalysisDetail>(`/api/analyses/${id}`)
      setDetail(result)
      if (result.analysis?.patch_batch?.finding_ids) {
        setSelected(new Set(result.analysis.patch_batch.finding_ids))
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : '분석 정보를 불러오지 못했습니다.')
    }
  }, [id])

  useEffect(() => {
    load()
  }, [load])

  useEffect(() => {
    const status = detail?.job.status
    if (!status || !['uploading', 'queued', 'analyzing'].includes(status)) return

    const timer = window.setInterval(load, 1800)
    return () => window.clearInterval(timer)
  }, [detail?.job.status, load])

  const findings = detail?.analysis?.findings ?? []
  const validated = useMemo(
    () => findings.filter((x) => x.validation.verdict === 'validated'),
    [findings]
  )

  const proposePatch = async () => {
    if (!id || !selected.size) return
    setPatchBusy(true)
    setError('')
    try {
      await api.post<PatchBatch>(`/api/analyses/${id}/patches/proposal`, {
        findingIds: Array.from(selected)
      })
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : '패치 생성에 실패했습니다.')
    } finally {
      setPatchBusy(false)
    }
  }

  const patchAction = async (action: 'approve' | 'reject') => {
    const patch = detail?.analysis?.patch_batch
    if (!id || !patch) return
    setPatchBusy(true)
    try {
      await api.post<PatchBatch>(
        `/api/analyses/${id}/patches/${encodeURIComponent(patch.patch_id)}/${action}`
      )
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : '패치 처리에 실패했습니다.')
    } finally {
      setPatchBusy(false)
    }
  }

  if (!detail) {
    return (
      <div className="page">
        <Link className="back-link" to="/analyses"><ArrowLeft size={16} /> 분석 이력</Link>
        {error ? <div className="error-box">{error}</div> : <div className="loader" />}
      </div>
    )
  }

  const { job, analysis } = detail
  const patch = analysis?.patch_batch

  return (
    <div className="page">
      <Link className="back-link" to="/analyses">
        <ArrowLeft size={16} />
        분석 이력
      </Link>

      <header className="detail-header">
        <div>
          <span className="eyebrow">SECURITY ANALYSIS</span>
          <h1>{job.projectName}</h1>
          <p>{new Date(job.createdAt).toLocaleString('ko-KR')} · {job.fileCount}개 파일</p>
        </div>
        <div className="detail-actions">
          <StatusBadge status={job.status} />
          {job.status === 'completed' && (
            <a
              className="secondary-button"
              href={api.downloadUrl(`/api/analyses/${job.id}/download`)}
            >
              <Download size={17} />
              프로젝트 다운로드
            </a>
          )}
        </div>
      </header>

      {error && <div className="error-box">{error}</div>}

      {['uploading', 'queued', 'analyzing'].includes(job.status) && (
        <section className="panel progress-panel">
          <div className="panel-head">
            <div>
              <h2>{job.message || '분석을 진행하고 있습니다.'}</h2>
              <p>페이지를 닫아도 분석 기록은 계정에 남습니다.</p>
            </div>
            <strong>{job.progress}%</strong>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${job.progress}%` }} />
          </div>
        </section>
      )}

      {job.status === 'failed' && (
        <section className="panel danger-panel">
          <XCircle />
          <div>
            <h2>분석에 실패했습니다.</h2>
            <p>{job.errorMessage || job.message}</p>
          </div>
        </section>
      )}

      {analysis && (
        <>
          <section className="stat-grid detail-stats">
            <MiniStat icon={<FileCode2 />} label="분석 소스" value={analysis.summary.source_file_count} />
            <MiniStat icon={<ShieldAlert />} label="발견 취약점" value={analysis.summary.finding_count} />
            <MiniStat icon={<CheckCircle2 />} label="검증됨" value={job.validatedFindingCount} />
            <MiniStat
              icon={<WandSparkles />}
              label="API 비용"
              value={`$${Number(analysis.summary.total_cost || 0).toFixed(4)}`}
            />
          </section>

          <section className="panel">
            <div className="panel-head">
              <div>
                <h2>취약점 결과</h2>
                <p>
                  검증된 항목을 선택하면 하나의 통합 패치 요청으로 수정안을 생성합니다.
                </p>
              </div>
              {!patch && (
                <button
                  className="primary-button compact"
                  disabled={!selected.size || patchBusy}
                  onClick={proposePatch}
                >
                  <WandSparkles size={16} />
                  {patchBusy ? '생성 중…' : `통합 패치 생성 (${selected.size})`}
                </button>
              )}
            </div>

            <div className="finding-list">
              {findings.length === 0 ? (
                <div className="empty-state">탐지된 취약점이 없습니다.</div>
              ) : (
                findings.map((bundle) => (
                  <FindingCard
                    key={bundle.finding.finding_id}
                    bundle={bundle}
                    selected={selected.has(bundle.finding.finding_id)}
                    disabled={Boolean(patch)}
                    onToggle={(checked) => {
                      setSelected((before) => {
                        const next = new Set(before)
                        if (checked) next.add(bundle.finding.finding_id)
                        else next.delete(bundle.finding.finding_id)
                        return next
                      })
                    }}
                  />
                ))
              )}
            </div>
          </section>

          {patch && (
            <section className="panel patch-panel">
              <div className="panel-head">
                <div>
                  <span className="eyebrow">PATCH PROPOSAL</span>
                  <h2>{patch.finding_ids.length}개 취약점 통합 수정안</h2>
                </div>
                <StatusBadge status={patch.status} />
              </div>

              <p className="patch-summary">{patch.summary}</p>
              <pre className="diff-view">{patch.unified_diff}</pre>

              {patch.status === 'proposed' && (
                <div className="patch-actions">
                  <button
                    className="primary-button compact"
                    disabled={patchBusy}
                    onClick={() => patchAction('approve')}
                  >
                    <CheckCircle2 size={16} />
                    승인 및 적용
                  </button>
                  <button
                    className="danger-button compact"
                    disabled={patchBusy}
                    onClick={() => patchAction('reject')}
                  >
                    <XCircle size={16} />
                    거절
                  </button>
                </div>
              )}
            </section>
          )}
        </>
      )}
    </div>
  )
}

function MiniStat({
  icon,
  label,
  value
}: {
  icon: React.ReactNode
  label: string
  value: string | number
}) {
  return (
    <article className="stat-card">
      <div className="stat-icon">{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  )
}

function FindingCard({
  bundle,
  selected,
  disabled,
  onToggle
}: {
  bundle: FindingBundle
  selected: boolean
  disabled: boolean
  onToggle: (checked: boolean) => void
}) {
  const { finding, validation } = bundle
  const isValidated = validation.verdict === 'validated'

  return (
    <article className="finding-card">
      <div className="finding-top">
        <div className="finding-title-wrap">
          <StatusBadge status={validation.verdict} />
          <h3>{finding.title}</h3>
          <span className="location">
            {finding.file}:{finding.line_start}-{finding.line_end}
            {finding.function ? ` · ${finding.function}` : ''}
          </span>
        </div>

        <div className="confidence">
          <span>신뢰도</span>
          <strong>{Math.round((validation.confidence || 0) * 100)}%</strong>
        </div>
      </div>

      <div className="badge-row">
        {(finding.cwes || []).map((cwe) => (
          <span className="tech-badge" key={cwe}>{cwe}</span>
        ))}
        {(finding.supporting_experts || (finding.expert ? [finding.expert] : []))
          .map((expert) => (
            <span className="tech-badge secondary" key={expert}>{expertLabel(expert)}</span>
          ))}
      </div>

      <div className="finding-grid">
        <div>
          <span className="field-label">원인</span>
          <p>{finding.root_cause}</p>
        </div>
        <div>
          <span className="field-label">영향</span>
          <p>{finding.consequence}</p>
        </div>
      </div>

      {(validation.reasons?.length ?? 0) > 0 && (
        <details>
          <summary>검증 근거 보기</summary>
          <ul>
            {validation.reasons?.map((reason, index) => (
              <li key={`${reason}-${index}`}>{reason}</li>
            ))}
          </ul>
        </details>
      )}

      {isValidated && (
        <label className="patch-selector">
          <input
            type="checkbox"
            checked={selected}
            disabled={disabled}
            onChange={(e) => onToggle(e.target.checked)}
          />
          통합 패치에 포함
        </label>
      )}
    </article>
  )
}

function expertLabel(expert: string) {
  const labels: Record<string, string> = {
    memory_bounds: 'E1 Memory Safety',
    lifetime_resource: 'E2 Lifetime / Resource',
    integer_size_type: 'E3 Integer / Size / Type',
    taint_api_contract: 'E4 Taint / API Contract',
    control_state_error: 'E5 Control / State / Error',
    concurrency_toctou: 'E6 Concurrency / TOCTOU'
  }
  return labels[expert] || expert
}
