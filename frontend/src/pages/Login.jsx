import { useState } from 'react'
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext.jsx'

export default function Login() {
  const { user, loading, login } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [form, setForm] = useState({ email: '', password: '' })
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  if (loading) return <div className="center-screen"><div className="spinner" /></div>
  if (user) return <Navigate to="/dashboard" replace />

  function change(e) {
    setForm({ ...form, [e.target.name]: e.target.value })
  }

  async function submit(e) {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      await login(form)
      navigate(location.state?.from || '/dashboard', { replace: true })
    } catch (err) {
      setError(err.message || 'Could not sign in')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-head">
          <div className="brand">
            <div className="brand-mark">NG</div>
            <div>
              <div className="brand-name">NeuroGuard</div>
              <div className="brand-sub">Cognitive Screening</div>
            </div>
          </div>
          <h2>Welcome back</h2>
          <p className="muted small mt-1">Sign in to continue your monitoring</p>
        </div>

        <form className="card" onSubmit={submit}>
          {error && <div className="alert alert-error">{error}</div>}

          <div className="field">
            <label className="label" htmlFor="email">Email</label>
            <input
              id="email" name="email" type="email" className="input" required
              autoComplete="email" value={form.email} onChange={change}
              placeholder="you@example.com"
            />
          </div>

          <div className="field">
            <div className="label-row">
              <label className="label" htmlFor="password">Password</label>
              <Link className="small" to="/forgot-password">Forgot password?</Link>
            </div>
            <input
              id="password" name="password" type="password" className="input" required
              autoComplete="current-password" value={form.password} onChange={change}
              placeholder="••••••••"
            />
          </div>

          <button className="btn btn-primary btn-block btn-lg mt-1" disabled={busy}>
            {busy ? 'Signing in…' : 'Sign in'}
          </button>
        </form>

        <div className="auth-foot">
          Don't have an account? <Link to="/signup">Create one</Link>
        </div>
      </div>
    </div>
  )
}
