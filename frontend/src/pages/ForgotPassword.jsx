import { useState } from 'react'
import { Link, Navigate } from 'react-router-dom'
import { api } from '../api/client.js'
import { useAuth } from '../context/AuthContext.jsx'

/**
 * Request a password reset link.
 *
 * The success screen is shown for ANY submitted address, including ones with no
 * account. That mirrors the API, which returns an identical response either way
 * so the endpoint cannot be used to discover who is registered. Saying "no
 * account found" here would leak exactly what the backend refuses to.
 */
export default function ForgotPassword() {
  const { user, loading } = useAuth()
  const [email, setEmail] = useState('')
  const [sent, setSent] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  if (loading) return <div className="center-screen"><div className="spinner" /></div>
  if (user) return <Navigate to="/dashboard" replace />

  async function submit(e) {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      await api.forgotPassword(email.trim().toLowerCase())
      setSent(true)
    } catch (err) {
      // Only reachable on a network failure or the rate limiter, never on
      // "unknown address" — that path returns 200 like any other.
      setError(err.message || 'Could not send the reset link')
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
          <h2>{sent ? 'Check your email' : 'Reset your password'}</h2>
          <p className="muted small mt-1">
            {sent
              ? 'If an account exists for that address, a reset link is on its way.'
              : 'Enter your email and we will send you a link to set a new password.'}
          </p>
        </div>

        {sent ? (
          <div className="card">
            <div className="alert alert-success">
              The link expires in 30 minutes and can only be used once. If it does
              not arrive, check your spam folder or request another.
            </div>
            <button
              className="btn btn-block mt-1"
              onClick={() => {
                setSent(false)
                setEmail('')
              }}
            >
              Use a different address
            </button>
          </div>
        ) : (
          <form className="card" onSubmit={submit}>
            {error && <div className="alert alert-error">{error}</div>}

            <div className="field">
              <label className="label" htmlFor="email">Email</label>
              <input
                id="email" name="email" type="email" className="input" required
                autoComplete="email" value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
              />
            </div>

            <button className="btn btn-primary btn-block btn-lg mt-1" disabled={busy}>
              {busy ? 'Sending…' : 'Send reset link'}
            </button>
          </form>
        )}

        <div className="auth-foot">
          Remembered it? <Link to="/login">Back to sign in</Link>
        </div>
      </div>
    </div>
  )
}
