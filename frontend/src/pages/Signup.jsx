import { useState } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import { GlassFilter, GlassPanel } from '../components/LiquidGlass.jsx'
import { useAuth } from '../context/AuthContext.jsx'

export default function Signup() {
  const { user, loading, signup } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({
    full_name: '', email: '', password: '', age: '', education_years: '',
  })
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
      await signup({
        full_name: form.full_name,
        email: form.email,
        password: form.password,
        age: Number(form.age),
        education_years:
          form.education_years === '' ? null : Number(form.education_years),
      })
      navigate('/dashboard', { replace: true })
    } catch (err) {
      setError(err.message || 'Could not create the account')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth-page bloom-scope gradient-bloom-field">
      <GlassFilter />
      <div className="auth-card">
        <div className="auth-head">
          <div className="brand">
            <div className="brand-mark">NG</div>
            <div>
              <div className="brand-name">NeuroGuard</div>
              <div className="brand-sub">Cognitive Screening</div>
            </div>
          </div>
          <h2>Create your account</h2>
          <p className="muted small mt-1">
            Age and education are used to adjust your scores against the right
            reference group
          </p>
        </div>

        <GlassPanel className="auth-glass">
        <form onSubmit={submit}>
          {error && <div className="alert alert-error">{error}</div>}

          <div className="field">
            <label className="label" htmlFor="full_name">Full name</label>
            <input
              id="full_name" name="full_name" className="input" required
              autoComplete="name" value={form.full_name} onChange={change}
            />
          </div>

          <div className="field">
            <label className="label" htmlFor="email">Email</label>
            <input
              id="email" name="email" type="email" className="input" required
              autoComplete="email" value={form.email} onChange={change}
            />
          </div>

          <div className="field">
            <label className="label" htmlFor="password">Password</label>
            <input
              id="password" name="password" type="password" className="input" required
              minLength={8} autoComplete="new-password"
              value={form.password} onChange={change}
            />
            <span className="hint">
              At least 8 characters, using both letters and numbers
            </span>
          </div>

          <div className="grid grid-2" style={{ gap: 14 }}>
            <div className="field">
              <label className="label" htmlFor="age">Age</label>
              <input
                id="age" name="age" type="number" className="input" required
                min={18} max={120} value={form.age} onChange={change}
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="education_years">
                Years of education
              </label>
              <input
                id="education_years" name="education_years" type="number"
                className="input" min={0} max={30}
                value={form.education_years} onChange={change}
                placeholder="Optional"
              />
            </div>
          </div>

          <button className="btn btn-primary btn-block btn-lg mt-1" disabled={busy}>
            {busy ? 'Creating account…' : 'Create account'}
          </button>
        </form>
        </GlassPanel>

        <div className="auth-foot">
          Already registered? <Link to="/login">Sign in</Link>
        </div>
      </div>
    </div>
  )
}
