import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'

/**
 * Connectivity pathway: resting-state fMRI (+ DTI connectome) -> brain graph ->
 * Graph Attention Network -> Normal / MCI / Dementia.
 *
 * Like the MRI page, the result is reported on its own, never folded into
 * the behavioural score. Until a model has been trained on real data the
 * page says so plainly instead of showing anything that looks like a result.
 */

const MAX_FMRI_MB = 400
const CLASS_COLORS = { Normal: 'var(--low)', MCI: 'var(--borderline)', Dementia: 'var(--elevated)', Impaired: 'var(--elevated)' }

function Chord({ edges }) {
  // Simple connectome view: regions on a circle, attended pairs as chords.
  const regions = [...new Set(edges.flatMap((e) => [e.region_a, e.region_b]))]
  const size = 320
  const c = size / 2
  const r = c - 70
  const pos = Object.fromEntries(
    regions.map((name, i) => {
      const a = (2 * Math.PI * i) / regions.length - Math.PI / 2
      return [name, { x: c + r * Math.cos(a), y: c + r * Math.sin(a), a }]
    })
  )
  const weight = (e) => e.importance ?? e.attention
  const max = Math.max(...edges.map(weight), 1e-9)
  return (
    <svg viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Most-attended region pairs"
      style={{ width: '100%', maxWidth: 360, display: 'block', margin: '0 auto' }}>
      {edges.map((e, i) => {
        const p = pos[e.region_a]
        const q = pos[e.region_b]
        return (
          <path key={i} d={`M ${p.x} ${p.y} Q ${c} ${c} ${q.x} ${q.y}`} fill="none"
            stroke="var(--brand)" strokeOpacity={0.25 + 0.75 * (weight(e) / max)}
            strokeWidth={1 + 3 * (weight(e) / max)} />
        )
      })}
      {regions.map((name) => {
        const p = pos[name]
        const right = Math.cos(p.a) >= 0
        return (
          <g key={name}>
            <circle cx={p.x} cy={p.y} r={4} fill="var(--text)" />
            <text x={p.x + (right ? 8 : -8)} y={p.y + 3} fontSize="8" fill="var(--text-muted)"
              textAnchor={right ? 'start' : 'end'}>
              {name.length > 22 ? name.slice(0, 21) + '…' : name}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

function FilePicker({ label, hint, accept, file, onFile, icon }) {
  const ref = useRef(null)
  return (
    <div className="field">
      <label className="label">{label}</label>
      <input ref={ref} type="file" accept={accept} style={{ display: 'none' }}
        onChange={(e) => { onFile(e.target.files?.[0] || null); e.target.value = '' }} />
      {!file ? (
        <div className="dropzone" role="button" tabIndex={0} onClick={() => ref.current?.click()}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); ref.current?.click() } }}>
          <div className="dropzone-icon">⇪</div>
          <div className="dropzone-title">Choose a file</div>
          <div className="dropzone-hint">{hint}</div>
        </div>
      ) : (
        <div className="file-chip">
          <div className="file-chip-icon">{icon}</div>
          <div style={{ minWidth: 0 }}>
            <div className="file-chip-name">{file.name}</div>
            <div className="file-chip-meta">{(file.size / (1024 * 1024)).toFixed(1)} MB</div>
          </div>
          <button className="file-chip-x" aria-label="Remove file" onClick={() => onFile(null)}>✕</button>
        </div>
      )}
    </div>
  )
}

export default function GraphAnalysis() {
  const [info, setInfo] = useState(null)
  const [fmri, setFmri] = useState(null)
  const [sc, setSc] = useState(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  function loadInfo() {
    setInfo(null)
    api.graphInfo().then(setInfo).catch(() => setInfo({ available: false, error: true }))
  }
  useEffect(loadInfo, [])

  function pickFmri(f) {
    setError('')
    if (f && !/\.nii(\.gz)?$/i.test(f.name)) {
      setError('The fMRI must be a 4-D .nii or .nii.gz file.')
      return
    }
    if (f && f.size > MAX_FMRI_MB * 1024 * 1024) {
      setError(`That file is larger than ${MAX_FMRI_MB} MB.`)
      return
    }
    setFmri(f)
  }

  function pickSc(f) {
    setError('')
    if (f && !/\.(csv|npy)$/i.test(f.name)) {
      setError('The connectivity matrix must be a .csv or .npy file.')
      return
    }
    setSc(f)
  }

  async function analyze() {
    setBusy(true)
    setError('')
    try {
      const fd = new FormData()
      fd.append('fmri', fmri, fmri.name)
      if (sc) fd.append('sc', sc, sc.name)
      setResult(await api.graphAnalyze(fd))
    } catch (err) {
      setError(err.status ? err.message : 'The analysis service could not be reached — try again in a moment.')
    } finally {
      setBusy(false)
    }
  }

  if (info === null) {
    return <div className="text-center" style={{ paddingTop: 60 }}><div className="spinner" /></div>
  }

  if (info.error) {
    return (
      <div className="card"><div className="empty">
        <div className="empty-icon">⚠</div>
        <h3>Couldn't reach the analysis service</h3>
        <button className="btn btn-primary mt-2" onClick={loadInfo}>Try again</button>
      </div></div>
    )
  }

  if (!info.available) {
    return (
      <div className="card"><div className="empty">
        <div className="empty-icon">⟡</div>
        <h3>{info.enabled === false ? 'Connectivity analysis is not available on this deployment'
          : 'Not available yet: the graph model has not been trained'}</h3>
        <p className="muted small mt-1" style={{ maxWidth: 560, margin: '8px auto 0' }}>
          {info.enabled === false
            ? 'This research feature is switched off here. The rest of NeuroGuard works normally.'
            : 'The pipeline is built, but no model has been trained on real resting-state fMRI and DTI data yet, so there are no results or accuracy figures to show. Nothing on this page is simulated.'}
        </p>
      </div></div>
    )
  }

  const cv = info.cv || {}
  const base = info.baseline || {}
  const pct = (m) => (m?.mean == null ? '—' : `${Math.round(m.mean * 100)}%`)
  const num = (m) => (m?.mean == null ? '—' : m.mean.toFixed(2))

  return (
    <div className="grid" style={{ gap: 20 }}>
      <div className="card">
        <div className="card-title">Analyse brain connectivity (research)</div>
        <div className="card-sub">
          Upload a resting-state fMRI series that has already been preprocessed into MNI space
          (for example an fMRIPrep output){info.graph?.with_sc ? ', plus the structural connectivity matrix from DTI tractography' : ''}.
          The brain is divided into {info.dataset?.regions ?? 'atlas'} regions, linked by how strongly their
          activity moves together, and a graph attention network classifies the network.
        </div>
        <div className="grid grid-2" style={{ gap: 14, maxWidth: 760 }}>
          <FilePicker label="Resting-state fMRI (4-D)" hint=".nii / .nii.gz, MNI space"
            accept=".nii,.nii.gz" file={fmri} onFile={pickFmri} icon="BOLD" />
          <FilePicker label={`Structural connectivity${info.graph?.with_sc ? '' : ' (optional)'}`}
            hint=".csv or .npy, one row per region" accept=".csv,.npy" file={sc} onFile={pickSc} icon="SC" />
        </div>
        <button className="btn btn-primary mt-2" disabled={!fmri || busy} onClick={analyze}>
          {busy ? 'Analysing… (this can take a minute)' : 'Analyse connectivity'}
        </button>
        <div aria-live="polite">{error && <div className="alert alert-error mt-3">{error}</div>}</div>
      </div>

      {result && (
        <div className="grid grid-2">
          <div className="card">
            <div className="card-title">Screening result</div>
            <div className="card-sub">Connectivity graph · {result.n_regions} regions{result.used_sc ? ' · FC + SC' : ' · FC only'}</div>
            <div style={{ color: CLASS_COLORS[result.predicted_class], fontSize: '1.6rem', fontWeight: 600, margin: '6px 0 8px' }}>
              {result.predicted_class}
            </div>
            <span className="badge badge-neutral">screening indicator — not a diagnosis</span>
            <div className="mt-3">
              {Object.entries(result.probabilities).map(([name, p]) => (
                <div key={name} className="prob-row">
                  <div className="prob-head"><span>{name}</span><span className="mono">{(p * 100).toFixed(1)}%</span></div>
                  <div className="prob-track">
                    <div className="prob-fill" style={{ width: `${Math.max(2, p * 100)}%`, background: CLASS_COLORS[name] }} />
                  </div>
                </div>
              ))}
            </div>
            <div className="disclaimer mt-3">{result.note}</div>
          </div>

          <div className="card">
            <div className="card-title">
              {result.model_type === 'logreg' ? 'Connections that drove this result' : 'Where the model looked'}
            </div>
            <div className="card-sub">
              {result.model_type === 'logreg'
                ? 'Region pairs whose connection strength pushed the score most towards this result. They show what drove the score, not which connection is abnormal.'
                : 'Region pairs with the highest attention. Attention shows what the model weighed most, not which connection is abnormal.'}
            </div>
            <Chord edges={result.top_edges} />
            <ol className="small" style={{ paddingLeft: 18, margin: '12px 0 0', lineHeight: 1.8 }}>
              {result.top_edges.map((e, i) => (
                <li key={i}>
                  {e.region_a} ↔ {e.region_b}{' '}
                  <span className="muted mono">
                    {result.model_type === 'logreg' ? 'weight' : 'attention'} {(e.importance ?? e.attention).toFixed(3)} · r {e.fc.toFixed(2)}
                  </span>
                </li>
              ))}
            </ol>
          </div>
        </div>
      )}

      <div className="grid grid-2">
        <div className="card">
          <div className="card-title">About this model</div>
          <div className="card-sub">{info.model}</div>
          <ul className="small muted" style={{ paddingLeft: 18, margin: 0, lineHeight: 1.9 }}>
            <li>{info.dataset?.name} · {info.dataset?.subjects} subjects · {info.dataset?.sessions} sessions</li>
            <li>{info.dataset?.split}</li>
            <li>Atlas: {info.dataset?.atlas} ({info.dataset?.regions} regions)</li>
          </ul>
          <div className="grid grid-3 mt-3" style={{ gap: 12 }}>
            <div><div className="stat-value">{pct(cv.accuracy)}</div><div className="stat-label">accuracy ({cv.folds}-fold CV)</div></div>
            <div><div className="stat-value">{num(cv.macro_f1)}</div><div className="stat-label">macro-F1</div></div>
            <div><div className="stat-value">{num(cv.auc)}</div><div className="stat-label">AUC</div></div>
          </div>
          <p className="tiny muted mt-2">
            {info.model_type === 'logreg'
              ? <>The graph attention network was also trained on the same folds and scored lower (accuracy {pct(info.gat_cv?.accuracy)}, AUC {num(info.gat_cv?.auc)}), so the simpler model is the one deployed.</>
              : <>Baseline (logistic regression on the same folds): accuracy {pct(base.accuracy)}, macro-F1 {num(base.macro_f1)}, AUC {num(base.auc)}.</>}
            {info.dataset?.majority_class_accuracy != null && <> Always guessing the most common class would score {pct({ mean: info.dataset.majority_class_accuracy })}.</>}
          </p>
        </div>
        <div className="card">
          <div className="card-title">Known limitations</div>
          <div className="card-sub">Stated up front, not buried</div>
          <ul className="small muted" style={{ paddingLeft: 18, margin: 0, lineHeight: 1.9 }}>
            {(info.caveats || []).map((c, i) => <li key={i}>{c}</li>)}
          </ul>
        </div>
      </div>

      <div className="disclaimer">
        The connectivity result and the behavioural risk score are reported separately on purpose:
        no dataset pairs these scans with the app's six tests, so combining them into one number
        would need a made-up weighting. Research and educational use only.
      </div>
    </div>
  )
}
