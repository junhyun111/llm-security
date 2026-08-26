import { FolderOpen, ScanSearch, UploadCloud } from 'lucide-react'
import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import type { AnalysisJob } from '../types'

export default function NewAnalysisPage() {
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const [files, setFiles] = useState<File[]>([])
  const [projectName, setProjectName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const openPicker = () => {
    if (inputRef.current) {
      inputRef.current.setAttribute('webkitdirectory', '')
      inputRef.current.setAttribute('directory', '')
      inputRef.current.click()
    }
  }

  const onFiles = (selected: FileList | null) => {
    const list = Array.from(selected || [])
    setFiles(list)

    const first = list[0] as File & { webkitRelativePath?: string }
    const root =
      first?.webkitRelativePath?.split('/')[0] ||
      first?.name ||
      'project'

    setProjectName(root)
    setError('')
  }

  const start = async () => {
    if (!files.length) return

    setBusy(true)
    setError('')

    const form = new FormData()
    form.append('project_name', projectName || 'project')

    for (const file of files) {
      const typed = file as File & { webkitRelativePath?: string }
      form.append('relative_paths', typed.webkitRelativePath || file.name)
      form.append('files', file, file.name)
    }

    try {
      const job = await api.postForm<AnalysisJob>('/api/analyses', form)
      navigate(`/analyses/${job.id}`)
    } catch (e) {
      setError(e instanceof Error ? e.message : '업로드에 실패했습니다.')
    } finally {
      setBusy(false)
    }
  }

  const totalBytes = files.reduce((sum, file) => sum + file.size, 0)

  return (
    <div className="page narrow-page">
      <header className="page-header">
        <div>
          <span className="eyebrow">NEW SECURITY SCAN</span>
          <h1>프로젝트 분석</h1>
          <p>C/C++ 프로젝트 폴더를 선택하면 기존 분석 엔진으로 안전성 검사를 시작합니다.</p>
        </div>
      </header>

      <section className="upload-card">
        <div className="upload-icon"><UploadCloud size={34} /></div>
        <h2>프로젝트 폴더 선택</h2>
        <p>
          .c, .cc, .cpp, .cxx, .h, .hh, .hpp 파일을 포함한 프로젝트 폴더를 선택하세요.
        </p>

        <input
          ref={inputRef}
          type="file"
          multiple
          hidden
          onChange={(e) => onFiles(e.target.files)}
        />

        <button className="secondary-button" onClick={openPicker}>
          <FolderOpen size={18} />
          폴더 선택
        </button>

        {files.length > 0 && (
          <div className="selection-card">
            <div>
              <span>선택된 프로젝트</span>
              <strong>{projectName}</strong>
            </div>
            <div>
              <span>파일</span>
              <strong>{files.length.toLocaleString()}개</strong>
            </div>
            <div>
              <span>용량</span>
              <strong>{formatBytes(totalBytes)}</strong>
            </div>
          </div>
        )}

        {error && <div className="error-box">{error}</div>}

        <button
          className="primary-button scan-button"
          disabled={!files.length || busy}
          onClick={start}
        >
          <ScanSearch size={18} />
          {busy ? '업로드 중…' : '보안 분석 시작'}
        </button>
      </section>

      <section className="info-strip">
        <strong>분석 과정</strong>
        <span>AST · CFG · Data Flow → Candidate Gate → Expert Router → 검증 → 결과 저장</span>
      </section>
    </div>
  )
}

function formatBytes(value: number) {
  if (!value) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const index = Math.min(
    Math.floor(Math.log(value) / Math.log(1024)),
    units.length - 1
  )
  return `${(value / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`
}
