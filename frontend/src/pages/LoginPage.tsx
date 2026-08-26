import { ShieldCheck } from 'lucide-react'
import { FormEvent, useState } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'

export default function LoginPage() {
  const { user, login } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  if (user) return <Navigate to="/" replace />

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setError('')
    setBusy(true)
    try {
      await login(email, password)
      navigate('/')
    } catch (e) {
      setError(e instanceof Error ? e.message : '로그인에 실패했습니다.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-glow" />
      <form className="auth-card" onSubmit={submit}>
        <div className="auth-logo"><ShieldCheck size={26} /></div>
        <div className="auth-head">
          <span className="eyebrow">SECURE CODE REVIEW</span>
          <h1>다시 만나서 반갑습니다.</h1>
          <p>C/C++ 프로젝트의 보안 분석 기록과 패치를 한곳에서 관리하세요.</p>
        </div>

        <label>
          이메일
          <input
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            required
          />
        </label>

        <label>
          비밀번호
          <input
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
            required
          />
        </label>

        {error && <div className="error-box">{error}</div>}

        <button className="primary-button" disabled={busy}>
          {busy ? '로그인 중…' : '로그인'}
        </button>

        <p className="auth-link">
          계정이 없나요? <Link to="/register">회원가입</Link>
        </p>
      </form>
    </div>
  )
}
