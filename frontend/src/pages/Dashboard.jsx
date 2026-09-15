import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { GlassFilter, GlassPanel } from '../components/LiquidGlass.jsx'
import RiskGauge from '../components/RiskGauge.jsx'
import { useAuth } from '../context/AuthContext.jsx'

const DOMAIN_LABEL = {
  memory: 'Memory',
  language: 'Language',
  executive: 'Executive',
  attention: 'Attention',
  working_memory: 'Working memory',
}

const TEST_BLURBS = [
  ['Memory recognition', 'Recall five objects after a filled delay'],
  ['Picture description', 'Describe a scene aloud — analysed by the trained model'],
  ['Trail Making B', 'Alternate numbers and letters under time pressure'],
  ['Serial sevens', 'Subtract 7 repeatedly from 100'],
  ['Digit span backward', 'Repeat digit sequences in reverse'],
  ['Verbal fluency', 'Name as many words as you can from one letter'],
]

export default function Dashboard() {
  const { user } = useAuth()
  const [summary, setSummary] = useState(null)
  const [trend, setTrend] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.summary(), api.trend()])
      .then(([s, t]) => {
        setSummary(s)
        setTrend(t)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return <div className="text-center" style={{ paddingTop: 60 }}><div className="spinner" /></div>
  }

  const latest = summary?.latest

  return (
    <div className="grid" style={{ gap: 20 }}>
      <GlassFilter />
      <div className="card bloom-scope gradient-bloom-field bloom-hero">
        <div className="row between wrap" style={{ gap: 20 }}>
          <div>
            <h2>Hello, {user?.full_name?.split(' ')[0]}</h2>
            <p className="muted small mt-1">
              {latest
                ? 'Here is where your last screening landed.'
                : 'You have not completed an assessment yet. It takes about ten minutes.'}
            </p>
          </div>
          <Link className="btn btn-primary btn-lg" to="/assessment">
            Start assessment
          </Link>
        </div>
      </div>

      {latest ? (
        <div className="grid grid-2">
          <GlassPanel className="lg-dark">
            <div className="card-title">Most recent result</div>
            <div className="card-sub">
              {new Date(latest.completed_at).toLocaleString()}
            </div>
            <div className="risk-hero">
              <RiskGauge score={latest.score_out_of_10} band={latest.band} size={140} />
              <div>
                <span className={`badge badge-${(latest.band || '').toLowerCase()}`}>
                  {latest.band}
                </span>
                <p className="muted small mt-2">
                  {latest.confidence} confidence
                </p>
                {latest.weakest_domain && (
                  <p className="small mt-1">
                    Weakest domain:{' '}
                    <strong>{DOMAIN_LABEL[latest.weakest_domain] || latest.weakest_domain}</strong>
                  </p>
                )}
                <Link className="btn btn-secondary mt-2" to={`/results/${latest.session_id}`}>
                  Full breakdown
                </Link>
              </div>
            </div>
          </GlassPanel>

          <GlassPanel className="lg-dark">
            <div className="card-title">Monitoring</div>
            <div className="card-sub">Change across your assessments</div>
            <div className="grid grid-2" style={{ gap: 14 }}>
              <div>
                <div className="stat-value">{summary.total_assessments}</div>
                <div className="stat-label">assessments taken</div>
              </div>
              <div>
                <div className="stat-value" style={{ textTransform: 'capitalize' }}>
                  {trend?.direction === 'insufficient_data' ? '—' : trend?.direction}
                </div>
                <div className="stat-label">
                  {trend?.change_since_first !== null && trend?.change_since_first !== undefined
                    ? `${trend.change_since_first > 0 ? '+' : ''}${trend.change_since_first} pts since first`
                    : 'take another to see a trend'}
                </div>
              </div>
            </div>
            <Link className="btn btn-secondary mt-3" to="/history">
              View history
            </Link>
          </GlassPanel>
        </div>
      ) : (
        <div className="card">
          <div className="empty">
            <div className="empty-icon">◔</div>
            <h3>No assessments yet</h3>
            <p className="muted small mt-1 mb-3">
              Your first assessment sets the baseline that later ones are
              compared against.
            </p>
            <Link className="btn btn-primary" to="/assessment">
              Take the first assessment
            </Link>
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-title">What the assessment covers</div>
        <div className="card-sub">
          Six tests, drawn from the Montreal Cognitive Assessment and the
          clinical picture-description task
        </div>
        <div className="grid grid-3">
          {TEST_BLURBS.map(([name, blurb], i) => (
            <div key={name}>
              <div className="row" style={{ gap: 8, alignItems: 'flex-start' }}>
                <span className="badge badge-neutral">{i + 1}</span>
                <div>
                  <div style={{ fontWeight: 600, fontSize: '.9rem' }}>{name}</div>
                  <div className="tiny muted">{blurb}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="disclaimer">
        NeuroGuard is a research and educational screening tool. It does not
        diagnose dementia, Alzheimer's disease or any other condition. If you
        are worried about your memory or thinking, speak to a doctor.
      </div>
    </div>
  )
}
