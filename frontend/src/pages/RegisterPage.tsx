import { ShieldCheck } from 'lucide-react'
import { FormEvent, useState } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'

export default function RegisterPage() {
  const { user, register } = useAuth()
  const navigate = useNavigate()
  const [displayName, setDisplayName] = useState('')
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
      await register(email, password, displayName)
      navigate('/')
    } catch (e) {
      setError(e instanceof Error ? e.message : '회원가입에 실패했습니다.')
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
          <span className="eyebrow">CREATE ACCOUNT</span>
          <h1>보안 분석을 시작하세요.</h1>
          <p>분석 프로젝트와 결과를 계정별로 안전하게 관리합니다.</p>
        </div>

        <label>
          이름
          <input
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            placeholder="표시할 이름"
            required
          />
        </label>

        <label>
          이메일
          <input
            type="email"
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
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="8자 이상, 영문 소문자 + 숫자"
            minLength={8}
            required
          />
        </label>

        {error && <div className="error-box">{error}</div>}

        <button className="primary-button" disabled={busy}>
          {busy ? '계정 생성 중…' : '회원가입'}
        </button>

        <p className="auth-link">
          이미 계정이 있나요? <Link to="/login">로그인</Link>
        </p>
      </form>
    </div>
  )
}
