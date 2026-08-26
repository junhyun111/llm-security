import {
  CheckCircle2,
  FileSearch,
  ShieldAlert,
  ShieldCheck,
  Wrench
} from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import JobTable from '../components/JobTable'
import type { Dashboard } from '../types'

const emptyDashboard: Dashboard = {
  totalScans: 0,
  completedScans: 0,
  totalFindings: 0,
  validatedFindings: 0,
  approvedPatches: 0,
  recentJobs: []
}

export default function DashboardPage() {
  const [data, setData] = useState<Dashboard>(emptyDashboard)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api
      .get<Dashboard>('/api/dashboard')
      .then(setData)
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <span className="eyebrow">OVERVIEW</span>
          <h1>Security Dashboard</h1>
          <p>프로젝트 보안 분석 현황과 최근 검수 기록을 확인합니다.</p>
        </div>
        <Link className="primary-button compact" to="/analyses/new">
          새 분석 시작
        </Link>
      </header>

      <section className="stat-grid">
        <Stat icon={<FileSearch />} label="전체 분석" value={data.totalScans} />
        <Stat icon={<CheckCircle2 />} label="완료 분석" value={data.completedScans} />
        <Stat icon={<ShieldAlert />} label="발견 취약점" value={data.totalFindings} />
        <Stat icon={<ShieldCheck />} label="검증된 취약점" value={data.validatedFindings} />
        <Stat icon={<Wrench />} label="승인 패치" value={data.approvedPatches} />
      </section>

      <section className="panel">
        <div className="panel-head">
          <div>
            <h2>최근 분석</h2>
            <p>가장 최근에 실행한 프로젝트 5개</p>
          </div>
          <Link className="text-link" to="/analyses">전체 보기</Link>
        </div>
        {loading ? <div className="skeleton-line" /> : <JobTable jobs={data.recentJobs} />}
      </section>
    </div>
  )
}

function Stat({
  icon,
  label,
  value
}: {
  icon: React.ReactNode
  label: string
  value: number
}) {
  return (
    <article className="stat-card">
      <div className="stat-icon">{icon}</div>
      <span>{label}</span>
      <strong>{value.toLocaleString()}</strong>
    </article>
  )
}
