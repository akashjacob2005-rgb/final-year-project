import { useState } from 'react'
import { Link, Navigate, useNavigate, useSearchParams } from 'react-router-dom'
import { api } from '../api/client.js'
import { useAuth } from '../context/AuthContext.jsx'

/**
 * Redeem a reset token and set a new password.
 *
 * The token arrives as ?token=... from the emailed link. It is never stored —
 * it goes straight from the query string into the one request that consumes it,
 * and the server marks it used. Resetting also revokes every existing session
 * for the account, so the user must sign in again afterwards; that is the point,
 * not an inconvenience.
 */
export default function ResetPassword() {
  const { user, loading } = useAuth()
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const token = params.get('token') || ''

  const [form, setForm] = useState({ password: '', confirm: '' })
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [done, setDone] = useState(false)

  if (loading) return <div className="center-screen"><div className="spinner" /></div>
  if (user) return <Navigate to="/dashboard" replace />

  function change(e) {
    setForm({ ...form, [e.target.name]: e.target.value })
  }

  async function submit(e) {
    e.preventDefault()
    setError('')
    if (form.password !== form.confirm) {
      setError('The two passwords do not match')
      return
    }
    setBusy(true)
    try {
      await api.resetPassword(token, form.password)
      setDone(true)
      setTimeout(() => navigate('/login', { replace: true }), 2500)
    } catch (err) {
      setError(err.message || 'Could not reset your password')
    } finally {
      setBusy(false)
    }
  }

  const head = (
    <div className="auth-head">
      <div className="brand">
        <div className="brand-mark">NG</div>
        <div>
          <div className="brand-name">NeuroGuard</div>
          <div className="brand-sub">Cognitive Screening</div>
        </div>
      </div>
      <h2>Choose a new password</h2>
    </div>
  )

  // A link opened without a token is malformed — say so immediately rather than
  // letting the user type a password and only then fail.
  if (!token) {
    return (
      <div className="auth-page">
        <div className="auth-card">
          {head}
          <div className="card">
            <div className="alert alert-error">
              This reset link is incomplete. Request a new one to continue.
            </div>
            <Link className="btn btn-primary btn-block mt-1" to="/forgot-password">
              Request a new link
            </Link>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        {head}

        {done ? (
          <div className="card">
            <div className="alert alert-success">
              Your password has been changed, and any other devices signed into
              this account have been signed out. Taking you to sign in…
            </div>
            <Link className="btn btn-block mt-1" to="/login">Sign in now</Link>
          </div>
        ) : (
          <form className="card" onSubmit={submit}>
            {error && <div className="alert alert-error">{error}</div>}

            <div className="field">
              <label className="label" htmlFor="password">New password</label>
              <input
                id="password" name="password" type="password" className="input" required
                minLength={8} autoComplete="new-password"
                value={form.password} onChange={change} placeholder="••••••••"
              />
              <div className="muted small mt-1">
                At least 8 characters, using both letters and numbers.
              </div>
            </div>

            <div className="field">
              <label className="label" htmlFor="confirm">Confirm new password</label>
              <input
                id="confirm" name="confirm" type="password" className="input" required
                minLength={8} autoComplete="new-password"
                value={form.confirm} onChange={change} placeholder="••••••••"
              />
            </div>

            <button className="btn btn-primary btn-block btn-lg mt-1" disabled={busy}>
              {busy ? 'Saving…' : 'Set new password'}
            </button>
          </form>
        )}

        <div className="auth-foot">
          <Link to="/login">Back to sign in</Link>
        </div>
      </div>
    </div>
  )
}
