const labels: Record<string, string> = {
  uploading: '업로드 중',
  queued: '대기',
  analyzing: '분석 중',
  completed: '완료',
  failed: '실패',
  validated: '검증됨',
  rejected: '기각',
  proposed: '제안됨',
  approved: '승인됨'
}

export default function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`status-badge status-${status}`}>
      <span className="status-dot" />
      {labels[status] || status}
    </span>
  )
}
