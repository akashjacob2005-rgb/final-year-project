import { useState } from 'react'
import { useAuth } from '../context/AuthContext.jsx'

export default function Profile() {
  const { user, updateProfile } = useAuth()
  const [form, setForm] = useState({
    full_name: user?.full_name ?? '',
    age: user?.age ?? '',
    education_years: user?.education_years ?? '',
  })
  const [status, setStatus] = useState({ type: '', message: '' })
  const [busy, setBusy] = useState(false)

  function change(e) {
    setForm({ ...form, [e.target.name]: e.target.value })
  }

  async function submit(e) {
    e.preventDefault()
    setBusy(true)
    setStatus({ type: '', message: '' })
    try {
      await updateProfile({
        full_name: form.full_name,
        age: Number(form.age),
        education_years:
          form.education_years === '' ? null : Number(form.education_years),
      })
      setStatus({ type: 'info', message: 'Profile updated.' })
    } catch (err) {
      setStatus({ type: 'error', message: err.message || 'Could not update profile' })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid" style={{ gap: 20, maxWidth: 620 }}>
      <form className="card" onSubmit={submit}>
        <div className="card-title">Your details</div>
        <div className="card-sub">
          Age and education decide which normative group your structured test
          scores are compared against, so keeping them accurate matters.
        </div>

        {status.message && (
          <div className={`alert alert-${status.type === 'error' ? 'error' : 'info'}`}>
            {status.message}
          </div>
        )}

        <div className="field">
          <label className="label" htmlFor="full_name">Full name</label>
          <input
            id="full_name" name="full_name" className="input"
            value={form.full_name} onChange={change} required
          />
        </div>

        <div className="field">
          <label className="label">Email</label>
          <input className="input" value={user?.email ?? ''} disabled />
          <span className="hint">Your email cannot be changed.</span>
        </div>

        <div className="grid grid-2" style={{ gap: 14 }}>
          <div className="field">
            <label className="label" htmlFor="age">Age</label>
            <input
              id="age" name="age" type="number" className="input"
              min={18} max={120} value={form.age} onChange={change} required
            />
          </div>
          <div className="field">
            <label className="label" htmlFor="education_years">Years of education</label>
            <input
              id="education_years" name="education_years" type="number"
              className="input" min={0} max={30}
              value={form.education_years ?? ''} onChange={change}
            />
          </div>
        </div>

        <button className="btn btn-primary" disabled={busy}>
          {busy ? 'Saving…' : 'Save changes'}
        </button>
      </form>

      <div className="disclaimer">
        Changing these values affects how <em>future</em> assessments are scored.
        Past results keep the age and education recorded at the time they were
        taken, so your history stays comparable.
      </div>
    </div>
  )
}
