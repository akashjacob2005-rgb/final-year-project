import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  CartesianGrid, Line, LineChart, ReferenceArea, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import { api } from '../api/client'

const DIRECTION_COPY = {
  worsening: ['Scores trending upward', 'Your screening score has risen since your first assessment. Repeat testing and, if it persists, a conversation with a doctor is the sensible next step.'],
  improving: ['Scores trending downward', 'Your screening score has fallen since your first assessment.'],
  stable: ['Stable', 'Your screening score has not changed meaningfully since your first assessment.'],
  insufficient_data: ['Not enough data yet', 'Take at least two assessments to see a trend.'],
}

export default function History() {
  const [trend, setTrend] = useState(null)
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  function load() {
    return Promise.all([api.trend(), api.sessions()])
      .then(([t, s]) => {
        setTrend(t)
        setSessions(s)
      })
      .catch((err) => setError(err.message || 'Could not load your history'))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    load()
  }, [])

  async function remove(id) {
    if (!window.confirm('Delete this assessment permanently?')) return
    try {
      await api.deleteSession(id)
      await load()
    } catch (err) {
      setError(err.message || 'Could not delete that assessment')
    }
  }

  if (loading) {
    return <div className="text-center" style={{ paddingTop: 60 }}><div className="spinner" /></div>
  }

  const points = (trend?.points || []).map((p) => ({
    ...p,
    label: new Date(p.completed_at).toLocaleDateString(undefined, {
      day: 'numeric', month: 'short',
    }),
  }))

  const [headline, blurb] = DIRECTION_COPY[trend?.direction] || DIRECTION_COPY.insufficient_data

  return (
    <div className="grid" style={{ gap: 20 }}>
      {error && <div className="alert alert-error">{error}</div>}

      <div className="card">
        <div className="card-title">{headline}</div>
        <div className="card-sub">{blurb}</div>

        {points.length === 0 ? (
          <div className="empty">
            <div className="empty-icon">◔</div>
            <p className="muted">No completed assessments yet.</p>
            <Link className="btn btn-primary mt-2" to="/assessment">
              Take one now
            </Link>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={points} margin={{ top: 10, right: 16, left: -14, bottom: 4 }}>
              {/* Band shading makes the y-axis meaningful at a glance. */}
              <ReferenceArea y1={0} y2={35} fill="var(--low)" fillOpacity={0.07} />
              <ReferenceArea y1={35} y2={65} fill="var(--borderline)" fillOpacity={0.07} />
              <ReferenceArea y1={65} y2={100} fill="var(--elevated)" fillOpacity={0.07} />
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="label" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
              <Tooltip
                formatter={(v, n) => [`${v}%`, n === 'risk_percent' ? 'Screening score' : n]}
                contentStyle={{ fontSize: 12, borderRadius: 8 }}
              />
              <Line
                type="monotone" dataKey="risk_percent" name="risk_percent"
                stroke="var(--brand)" strokeWidth={2.5}
                dot={{ r: 4, fill: 'var(--brand)' }} activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {sessions.length > 0 && (
        <div className="card">
          <div className="card-title">All assessments</div>
          <div className="card-sub">Most recent first</div>
          <div style={{ overflowX: 'auto' }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Score</th>
                  <th>Band</th>
                  <th>Confidence</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {sessions.map((s) => (
                  <tr key={s.session_id}>
                    <td>{new Date(s.completed_at).toLocaleString()}</td>
                    <td className="mono">{s.risk_percent}%</td>
                    <td>
                      <span className={`badge badge-${(s.band || '').toLowerCase()}`}>
                        {s.band}
                      </span>
                    </td>
                    <td className="muted">{s.confidence}</td>
                    <td>
                      <div className="row" style={{ justifyContent: 'flex-end' }}>
                        <Link className="btn btn-ghost small" to={`/results/${s.session_id}`}>
                          View
                        </Link>
                        <button className="btn btn-ghost small" onClick={() => remove(s.session_id)}>
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="disclaimer">
        A single score says relatively little. A consistent direction of travel
        across several assessments, taken under similar conditions, is far more
        informative — and is still not a diagnosis.
      </div>
    </div>
  )
}
