import { useEffect, useState } from 'react'
import { api } from '../api'
import JobTable from '../components/JobTable'
import type { AnalysisJob } from '../types'

export default function HistoryPage() {
  const [jobs, setJobs] = useState<AnalysisJob[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api
      .get<AnalysisJob[]>('/api/analyses')
      .then(setJobs)
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <span className="eyebrow">HISTORY</span>
          <h1>분석 이력</h1>
          <p>내 계정으로 실행한 모든 C/C++ 프로젝트 분석을 확인합니다.</p>
        </div>
      </header>

      <section className="panel">
        {loading ? <div className="skeleton-line" /> : <JobTable jobs={jobs} />}
      </section>
    </div>
  )
}
