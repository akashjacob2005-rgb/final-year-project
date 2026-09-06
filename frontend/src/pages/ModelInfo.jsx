import { useEffect, useState } from 'react'
import { api } from '../api/client'

/**
 * Transparency page. Deliberately shows the model's real cross-validated
 * numbers, including the models that lost, and states plainly which parts of
 * the pipeline are learned and which are not.
 */
export default function ModelInfo() {
  const [info, setInfo] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api.modelInfo().then(setInfo).catch((err) => setError(err.message))
  }, [])

  if (error) return <div className="alert alert-error">{error}</div>
  if (!info) {
    return <div className="text-center" style={{ paddingTop: 60 }}><div className="spinner" /></div>
  }

  if (!info.available) {
    return (
      <div className="card">
        <div className="empty">
          <div className="empty-icon">⚙</div>
          <h3>No trained model loaded</h3>
          <p className="muted small mt-1">
            Run <code>python ml/download_data.py</code> then{' '}
            <code>python ml/train_language_model.py</code>.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="grid" style={{ gap: 20 }}>
      <div className="card">
        <div className="card-title">What is actually machine-learned here</div>
        <div className="card-sub">
          Being precise about this matters more than the headline number
        </div>

        <table className="table">
          <tbody>
            <tr>
              <td style={{ width: 190 }}>
                <strong>Picture description</strong>
                <div className="tiny faint">Language pathway</div>
              </td>
              <td>
                <span className="badge badge-low">Trained model</span>
                <div className="tiny muted mt-1">
                  A classifier fitted on {info.dataset?.n_total} labelled clinical
                  transcripts and evaluated by cross-validation. The numbers
                  below are real.
                </div>
              </td>
            </tr>
            <tr>
              <td>
                <strong>The five other tests</strong>
                <div className="tiny faint">Structured pathway</div>
              </td>
              <td>
                <span className="badge badge-borderline">Rule-scored</span>
                <div className="tiny muted mt-1">
                  Scored by published MoCA rules, then expressed as z-scores
                  against age and education norms. No model is involved.
                </div>
              </td>
            </tr>
            <tr>
              <td>
                <strong>Final combination</strong>
                <div className="tiny faint">Fusion</div>
              </td>
              <td>
                <span className="badge badge-borderline">Fixed weights</span>
                <div className="tiny muted mt-1">
                  No public dataset exists of people taking these six tests with
                  diagnostic labels, so there is nothing to fit fusion weights
                  against. The weighting is a documented judgement, not a
                  learned parameter.
                </div>
              </td>
            </tr>
            <tr>
              <td>
                <strong>MRI analysis</strong>
                <div className="tiny faint">Structural pathway</div>
              </td>
              <td>
                <span className="badge badge-low">Trained model</span>
                <div className="tiny muted mt-1">
                  A ResNet-18 CNN fine-tuned on OASIS-3 brain scans labelled by
                  CDR stage, with a subject-wise split. Reported on its own
                  page and never fused into the behavioural score — see the MRI
                  Analysis page for its real metrics.
                </div>
              </td>
            </tr>
            <tr>
              <td>
                <strong>Speech timing</strong>
                <div className="tiny faint">Pauses, rate</div>
              </td>
              <td>
                <span className="badge badge-neutral">Descriptive only</span>
                <div className="tiny muted mt-1">
                  The training corpus is text with no audio, so these could not
                  be validated. They are displayed but excluded from the score.
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <div className="card-title">Model performance</div>
        <div className="card-sub">
          {info.evaluation}. The deployed model is highlighted.
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table className="table">
            <thead>
              <tr>
                <th>Configuration</th>
                <th>ROC-AUC</th>
                <th>95% CI</th>
                <th>Accuracy</th>
                <th>Sensitivity</th>
                <th>Specificity</th>
              </tr>
            </thead>
            <tbody>
              {info.models.map((m) => {
                const deployed = m.name === info.deployed_model
                return (
                  <tr key={m.name} style={deployed ? { background: 'var(--brand-soft)' } : undefined}>
                    <td>
                      {m.name}
                      {deployed && (
                        <span className="badge badge-low" style={{ marginLeft: 8 }}>
                          deployed
                        </span>
                      )}
                    </td>
                    <td className="mono"><strong>{m.roc_auc}</strong></td>
                    <td className="mono faint">
                      {m.roc_auc_ci95[0]}–{m.roc_auc_ci95[1]}
                    </td>
                    <td className="mono">{m.accuracy}</td>
                    <td className="mono">{m.sensitivity}</td>
                    <td className="mono">{m.specificity}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {info.deployed_because && (
          <div className="alert alert-info mt-3" style={{ marginBottom: 0 }}>
            <strong>Why this model was deployed rather than the top scorer:</strong>{' '}
            {info.deployed_because}.
          </div>
        )}
      </div>

      <div className="grid grid-2">
        <div className="card">
          <div className="card-title">Training data</div>
          <div className="card-sub">{info.dataset?.source}</div>
          <div className="grid grid-3" style={{ gap: 12 }}>
            <div>
              <div className="stat-value">{info.dataset?.n_total}</div>
              <div className="stat-label">transcripts</div>
            </div>
            <div>
              <div className="stat-value">{info.dataset?.n_dementia}</div>
              <div className="stat-label">dementia</div>
            </div>
            <div>
              <div className="stat-value">{info.dataset?.n_control}</div>
              <div className="stat-label">control</div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-title">Known limitations</div>
          <div className="card-sub">Stated up front, not buried</div>
          <ul className="small muted" style={{ paddingLeft: 18, margin: 0, lineHeight: 1.9 }}>
            {info.caveats.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </div>
      </div>

      <div className="disclaimer">
        Sensitivity is the proportion of true dementia cases the model flags;
        specificity is the proportion of healthy controls it correctly leaves
        alone. Neither is close to 100%, which is exactly why this is a
        screening indicator and not a diagnostic test.
      </div>
    </div>
  )
}
