import { ChevronRight } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import type { AnalysisJob } from '../types'
import StatusBadge from './StatusBadge'

export default function JobTable({
  jobs,
  emptyText = '아직 분석 기록이 없습니다.'
}: {
  jobs: AnalysisJob[]
  emptyText?: string
}) {
  const navigate = useNavigate()

  if (!jobs.length) {
    return <div className="empty-state">{emptyText}</div>
  }

  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th>프로젝트</th>
            <th>상태</th>
            <th>취약점</th>
            <th>검증됨</th>
            <th>분석일</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <tr key={job.id} onClick={() => navigate(`/analyses/${job.id}`)}>
              <td>
                <strong>{job.projectName}</strong>
                <span className="muted table-sub">{job.fileCount} files</span>
              </td>
              <td><StatusBadge status={job.status} /></td>
              <td>{job.findingCount}</td>
              <td>{job.validatedFindingCount}</td>
              <td>{new Date(job.createdAt).toLocaleString('ko-KR')}</td>
              <td className="table-arrow"><ChevronRight size={17} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
