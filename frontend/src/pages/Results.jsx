import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api } from '../api/client'
import RiskGauge from '../components/RiskGauge.jsx'

const DOMAIN_LABEL = {
  memory: 'Memory',
  language: 'Language',
  executive: 'Executive function',
  attention: 'Attention',
  working_memory: 'Working memory',
}

const TEST_LABEL = {
  memory_recognition: 'Memory recognition',
  picture_description: 'Picture description',
  trail_making_b: 'Trail Making B',
  serial_sevens: 'Serial sevens',
  digit_span_backward: 'Digit span backward',
  verbal_fluency: 'Verbal fluency',
}

const BAND_TEXT = {
  Low: 'Your performance is broadly in line with expectations for your age and education.',
  Borderline: 'Some results sit below expectation for your age and education. A single screening cannot tell you why — repeating it in a few weeks is the useful next step.',
  Elevated: 'Several results sit below expectation for your age and education. This is a screening signal, not a diagnosis. Consider discussing it with a doctor.',
}

/**
 * Put a component probability on the same /10 scale as the headline.
 *
 * The model emits P(decline), where high is bad. The page shows a score where
 * high is good. Displaying both directions side by side is how people misread a
 * result — "13%" next to "8.7/10" invites the reader to treat 13 as the score.
 * Mirrors fusion.fuse()["score_out_of_10"].
 */
function toScore(probability) {
  return ((1 - probability) * 10).toFixed(1)
}

function bandClass(band) {
  return `badge badge-${(band || 'neutral').toLowerCase()}`
}

export default function Results() {
  const { sessionId } = useParams()
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api
      .getResult(sessionId)
      .then(setResult)
      .catch((err) => setError(err.message || 'Could not load this result'))
  }, [sessionId])

  if (error) return <div className="alert alert-error">{error}</div>
  if (!result) {
    return (
      <div className="text-center" style={{ paddingTop: 60 }}>
        <div className="spinner" />
      </div>
    )
  }

  const domainData = Object.entries(result.domain_z || {}).map(([k, v]) => ({
    domain: DOMAIN_LABEL[k] || k,
    z: Number(v.toFixed(2)),
  }))

  const contributions = result.language?.contributions || []
  const maxContribution = Math.max(
    ...contributions.map((c) => Math.abs(c.contribution)),
    0.0001
  )
  const timing = result.language?.timing || {}
  const hasTiming = Object.keys(timing).length > 0 && timing.word_count > 0

  return (
    <div className="grid" style={{ gap: 20 }}>
      {/* ---------------------------------------------------- headline */}
      <div className="card">
        <div className="risk-hero">
          <RiskGauge score={result.score_out_of_10} band={result.band} />
          <div style={{ flex: 1, minWidth: 260 }}>
            <div className="row wrap mb-1">
              <span className={bandClass(result.band)}>{result.band}</span>
              <span className="badge badge-neutral">
                {result.confidence} confidence
              </span>
            </div>
            <h2 className="mb-1">Screening result</h2>
            <p className="muted small">{BAND_TEXT[result.band]}</p>

            {/* Without this the page contradicts itself: the band was raised by
                the safety override, so it sits above what the score alone would
                give. Explain it where the mismatch is actually seen. */}
            {result.band_escalated && (
              <p className="small mt-1">
                <strong>Why this band:</strong> your combined score is{' '}
                {result.score_out_of_10} out of 10, which on its own would read{' '}
                lower. One part of the assessment was far enough outside the
                expected range to raise it on its own.
              </p>
            )}

            {result.notes?.length > 0 && (
              <div className="alert alert-warn mt-2" style={{ marginBottom: 0 }}>
                {result.notes.map((n, i) => (
                  <div key={i}>{n}</div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ---------------------------------------------------- how it was made */}
      <div className="grid grid-2">
        <div className="card">
          <div className="card-title">How this score was produced</div>
          <div className="card-sub">
            Two components, each out of 10, combined with fixed published weights
          </div>
          <table className="table">
            <tbody>
              <tr>
                <td>
                  Language model
                  <div className="tiny faint">
                    Trained classifier · your picture description
                  </div>
                </td>
                <td className="mono text-center" style={{ width: 90 }}>
                  {result.components.language_probability === null
                    ? '—'
                    : `${toScore(result.components.language_probability)} / 10`}
                </td>
                <td className="mono faint text-center" style={{ width: 70 }}>
                  ×{result.weights.language}
                </td>
              </tr>
              <tr>
                <td>
                  Structured tests
                  <div className="tiny faint">
                    Age/education-adjusted composite z = {result.components.composite_z}
                  </div>
                </td>
                <td className="mono text-center">
                  {toScore(result.components.structured_probability)} / 10
                </td>
                <td className="mono faint text-center">
                  ×{result.weights.structured}
                </td>
              </tr>
            </tbody>
          </table>
          <p className="tiny faint mt-2">
            {result.method}. <Link to="/model">See how the model performs →</Link>
          </p>
        </div>

        <div className="card">
          <div className="card-title">Performance by cognitive domain</div>
          <div className="card-sub">
            z-scores against age and education norms · 0 is typical, below −1.5
            is the usual impairment threshold
          </div>
          {domainData.length === 0 ? (
            <p className="muted small">No domain scores available.</p>
          ) : (
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={domainData} margin={{ top: 6, right: 8, left: -18, bottom: 4 }}>
                <XAxis
                  dataKey="domain" tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                  interval={0} angle={-14} textAnchor="end" height={52}
                />
                <YAxis domain={[-3, 3]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                <Tooltip
                  formatter={(v) => [`z = ${v}`, 'Score']}
                  contentStyle={{ fontSize: 12, borderRadius: 8 }}
                />
                <ReferenceLine y={0} stroke="var(--text-faint)" />
                <ReferenceLine
                  y={-1.5} stroke="var(--elevated)" strokeDasharray="4 3"
                  label={{ value: 'impairment', fontSize: 9, fill: 'var(--elevated)', position: 'insideBottomRight' }}
                />
                <Bar dataKey="z" radius={[4, 4, 0, 0]}>
                  {domainData.map((d, i) => (
                    <Cell
                      key={i}
                      fill={d.z < -1.5 ? 'var(--elevated)' : d.z < -0.5 ? 'var(--borderline)' : 'var(--low)'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* ---------------------------------------------------- language detail */}
      {result.language && (
        <div className="grid grid-2">
          <div className="card">
            <div className="card-title">What the language model noticed</div>
            <div className="card-sub">
              The strongest signals in your description, and which way each
              pushed the score
            </div>

            {contributions.length === 0 ? (
              <p className="muted small">
                {result.language.reason ||
                  'No language analysis is available for this session.'}
              </p>
            ) : (
              <div>
                {contributions.map((c) => {
                  const up = c.contribution > 0
                  const width = (Math.abs(c.contribution) / maxContribution) * 100
                  return (
                    <div className="marker-row" key={c.feature}>
                      <span style={{ flex: 1 }}>{c.label}</span>
                      <div className="marker-bar">
                        <div
                          className={`marker-fill ${up ? 'up' : 'down'}`}
                          style={{
                            width: `${width / 2}%`,
                            left: up ? '50%' : `${50 - width / 2}%`,
                          }}
                        />
                      </div>
                      <span
                        className="tiny mono"
                        style={{ width: 92, textAlign: 'right', color: up ? 'var(--elevated)' : 'var(--low)' }}
                      >
                        {up ? 'raises' : 'lowers'}
                      </span>
                    </div>
                  )
                })}
                <p className="tiny faint mt-2">
                  Based on {result.language.word_count} words of transcript.
                </p>
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-title">Speech characteristics</div>
            <div className="card-sub">
              Measured from your recording — shown for information only
            </div>

            {!hasTiming ? (
              <p className="muted small">
                No audio timing available (typed response, or no recording made).
              </p>
            ) : (
              <>
                <div className="grid grid-2" style={{ gap: 12 }}>
                  <div>
                    <div className="stat-value" style={{ fontSize: '1.4rem' }}>
                      {timing.speech_rate_wpm}
                    </div>
                    <div className="stat-label">words per minute</div>
                  </div>
                  <div>
                    <div className="stat-value" style={{ fontSize: '1.4rem' }}>
                      {timing.pause_count}
                    </div>
                    <div className="stat-label">pauses</div>
                  </div>
                  <div>
                    <div className="stat-value" style={{ fontSize: '1.4rem' }}>
                      {timing.long_pause_count}
                    </div>
                    <div className="stat-label">pauses over 1s</div>
                  </div>
                  <div>
                    <div className="stat-value" style={{ fontSize: '1.4rem' }}>
                      {timing.duration_seconds}s
                    </div>
                    <div className="stat-label">total duration</div>
                  </div>
                </div>
                <div className="disclaimer mt-3">
                  These timing measures are <strong>not</strong> part of your
                  score. The model was trained on written transcripts with no
                  audio, so pause and rate features could not be validated. They
                  are reported as observations only.
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* ---------------------------------------------------- per-test table */}
      <div className="card">
        <div className="card-title">Individual test results</div>
        <div className="card-sub">Every test you completed in this session</div>
        <div style={{ overflowX: 'auto' }}>
          <table className="table">
            <thead>
              <tr>
                <th>Test</th>
                <th>Domain</th>
                <th>Raw</th>
                <th>Score</th>
                <th>z</th>
              </tr>
            </thead>
            <tbody>
              {result.tests.map((t) => (
                <tr key={t.id}>
                  <td>{TEST_LABEL[t.test_name] || t.test_name}</td>
                  <td className="muted">{DOMAIN_LABEL[t.domain] || t.domain}</td>
                  <td className="mono">
                    {t.raw_measure === null ? '—' : t.raw_measure.toFixed(2)}
                  </td>
                  <td className="mono">
                    {t.score === null ? '—' : `${t.score} / ${t.max_score}`}
                  </td>
                  <td className="mono">
                    {t.z_score === null ? (
                      <span className="faint">n/a</span>
                    ) : (
                      <span
                        style={{
                          color: t.z_score < -1.5 ? 'var(--elevated)' : t.z_score < -0.5 ? 'var(--borderline)' : 'var(--low)',
                        }}
                      >
                        {t.z_score > 0 ? '+' : ''}{t.z_score.toFixed(2)}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="disclaimer">{result.disclaimer}</div>

      <div className="row">
        <Link className="btn btn-secondary" to="/history">
          View monitoring history
        </Link>
        <Link className="btn btn-primary" to="/assessment">
          Take another assessment
        </Link>
      </div>
    </div>
  )
}
