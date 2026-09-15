import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'

/**
 * MRI pathway. Upload a T1-weighted NIfTI volume; the backend extracts the
 * same axial slices used in training (16 in the current model) and runs the
 * OASIS-3 CNN.
 *
 * Deliberately separate from the behavioural risk score: no dataset pairs MRI
 * with the six tests, so the two results are reported side by side, unfused.
 */

// First class is always "normal" (green); the last is the most severe (red);
// anything between is intermediate (amber). Works for binary and 3-class.
function classStyle(index, total) {
  if (index === 0) return 'badge-low'
  if (index === total - 1) return 'badge-elevated'
  return 'badge-borderline'
}

function classColor(index, total) {
  if (index === 0) return 'var(--low)'
  if (index === total - 1) return 'var(--elevated)'
  return 'var(--borderline)'
}

function verdictClass(index, total) {
  if (index === 0) return 'verdict-low'
  if (index === total - 1) return 'verdict-high'
  return 'verdict-mid'
}

// Turns the raw prediction into sentences a non-technical user can follow.
// Everything is derived from the result itself (class, confidence, gap to the
// runner-up), so the wording adapts to each uploaded scan.
function PlainExplanation({ result, fileName }) {
  const ranked = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
  const topP = ranked[0]?.[1] ?? result.confidence
  const runnerUpP = ranked[1]?.[1] ?? 0
  const impaired = result.predicted_index > 0
  const topPct = (topP * 100).toFixed(1)
  const runnerUpPct = (runnerUpP * 100).toFixed(1)

  const subject = !fileName
    ? 'your scan'
    : fileName.startsWith('the ')
      ? fileName // sample-scan label, already phrased
      : `your scan "${fileName}"`
  const what = `The AI examined ${result.slices_analysed} cross-section images ("slices") from ${subject} and averaged its opinion across all of them.`

  const found = impaired
    ? 'In this scan it noticed some patterns that it has learned to associate with possible memory-related brain changes.'
    : 'In this scan it did not find the patterns it has learned to associate with memory-related brain changes — to the model, this scan looks normal.'

  let sure
  if (topP >= 0.85) {
    sure = `The model is quite confident about this (${topPct}%), so the two possibilities were not a close call for this scan.`
  } else if (topP >= 0.65) {
    sure = `The model is moderately confident (${topPct}% versus ${runnerUpPct}% for the other possibility) — a reasonable indication, but not a certainty.`
  } else {
    sure = `However, the model is only ${topPct}% sure, versus ${runnerUpPct}% for the other possibility — close to a coin flip. This scan sits in the model's grey area, so the result should not be relied on either way.`
  }

  const advice = impaired
    ? 'A result like this is a reason to mention it to a doctor during a routine visit — not a cause for alarm, and not a diagnosis. Only a clinician with proper testing can assess this.'
    : 'Reassuring as this looks, it is a research tool and not a medical opinion — it cannot rule anything out. Any real concerns about memory should always go to a doctor.'

  return (
    <div className="explain-panel mt-3">
      <div style={{ fontWeight: 650, marginBottom: 6 }}>What does this mean?</div>
      <p style={{ margin: '0 0 8px' }}>{what} {found}</p>
      <p style={{ margin: '0 0 8px' }}>{sure}</p>
      <p style={{ margin: 0 }}>{advice}</p>
    </div>
  )
}

const HONESTY_TEXT =
  'One honest caveat: this model was trained on a research dataset with more impaired than healthy scans, and its percentages are not calibrated probabilities. In validation it flagged a meaningful share of healthy scans as impaired, so an "impairment" result here is a prompt to check properly — not a measured risk.'

function SliceViewer({ result }) {
  const impaired = result.predicted_index > 0
  const [showAttention, setShowAttention] = useState(false)
  const [zoomed, setZoomed] = useState(null)
  const closeBtnRef = useRef(null)
  const lastTriggerRef = useRef(null)

  // Escape closes the zoom modal; focus moves in on open and back on close.
  useEffect(() => {
    if (!zoomed) return
    closeBtnRef.current?.focus()
    const onKey = (e) => e.key === 'Escape' && setZoomed(null)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('keydown', onKey)
      lastTriggerRef.current?.focus()
    }
  }, [zoomed])

  const slices = result.slices || []
  if (!slices.length) return null

  // top-3 slices by the model's own per-slice confidence get flagged
  const top3 = new Set(
    [...slices]
      .sort((a, b) => b.class_probability - a.class_probability)
      .slice(0, 3)
      .map((s) => s.index)
  )
  const isHot = (s) => impaired && top3.has(s.index)

  return (
    <div className="card">
      <div
        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, flexWrap: 'wrap' }}
      >
        <div>
          <div className="card-title">Scan slices</div>
          <div className="card-sub">
            The {slices.length} axial slices the model analysed
            {impaired ? ' — outlined slices are where the model was most confident' : ''}
          </div>
        </div>
        <div className="toggle-pill" role="group" aria-label="Slice view mode">
          <button
            className={!showAttention ? 'active' : ''}
            aria-pressed={!showAttention}
            onClick={() => setShowAttention(false)}
          >
            Original
          </button>
          <button
            className={showAttention ? 'active' : ''}
            aria-pressed={showAttention}
            onClick={() => setShowAttention(true)}
          >
            Model attention
          </button>
        </div>
      </div>

      <div className="slice-grid" style={{ marginTop: 14 }}>
        {slices.map((s) => (
          <button
            key={s.index}
            className={`slice-tile ${isHot(s) ? 'hot' : ''}`}
            aria-label={`Enlarge axial slice ${s.index + 1}`}
            onClick={(e) => {
              lastTriggerRef.current = e.currentTarget
              setZoomed(s)
            }}
          >
            <img
              src={`data:image/png;base64,${showAttention ? s.overlay : s.image}`}
              alt={`Axial slice ${s.index + 1}`}
            />
            <span className="slice-tag">{s.index + 1}</span>
          </button>
        ))}
      </div>

      <div className="slice-caption">
        Click or tap any slice to enlarge it. The "Model attention" view highlights
        the regions that influenced the model most.
      </div>

      <div className="disclaimer mt-3">{result.explanation_note}</div>

      {zoomed && (
        <div className="slice-modal-backdrop" onClick={() => setZoomed(null)}>
          <div
            className="slice-modal"
            role="dialog"
            aria-modal="true"
            aria-label={`Axial slice ${zoomed.index + 1} enlarged`}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, gap: 12 }}>
              <div>
                <div className="card-title" style={{ margin: 0 }}>Slice {zoomed.index + 1}</div>
                <div className="tiny muted">
                  Model confidence for "{result.predicted_class}" on this slice:{' '}
                  {(zoomed.class_probability * 100).toFixed(1)}% — confidence, not severity
                </div>
              </div>
              <button ref={closeBtnRef} className="btn btn-secondary" onClick={() => setZoomed(null)}>
                Close
              </button>
            </div>
            <div className="pair">
              <figure>
                <img src={`data:image/png;base64,${zoomed.image}`} alt="Original slice" />
                <figcaption>Original (as analysed)</figcaption>
              </figure>
              <figure>
                <img src={`data:image/png;base64,${zoomed.overlay}`} alt="Model attention overlay" />
                <figcaption>Model attention (Grad-CAM)</figcaption>
              </figure>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

const MAX_MB = 80

export default function MriAnalysis() {
  const [info, setInfo] = useState(null) // null = loading; {available, error?}
  const [file, setFile] = useState(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [drag, setDrag] = useState(false)
  const inputRef = useRef(null)
  const resultRef = useRef(null)
  const dragDepth = useRef(0)

  function loadInfo() {
    setInfo(null)
    api
      .mriInfo()
      .then(setInfo)
      .catch(() => setInfo({ available: false, error: true }))
  }
  useEffect(loadInfo, [])

  // When a result arrives, take the user to it.
  useEffect(() => {
    if (result && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
      resultRef.current.focus({ preventScroll: true })
    }
  }, [result])

  function accept(f) {
    setError('')
    if (!f) return
    if (!/(\.nii(\.gz)?|\.zip)$/i.test(f.name)) {
      setFile(null)
      setError('Please choose your MRI as a .nii / .nii.gz file, or zip the scan folder from your MRI CD and upload the .zip.')
      return
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setFile(null)
      setError(`That file is ${(f.size / (1024 * 1024)).toFixed(0)} MB — the limit is ${MAX_MB} MB. A single T1w scan is usually 5–30 MB.`)
      return
    }
    setFile(f)
  }

  function pick(e) {
    accept(e.target.files?.[0])
    e.target.value = '' // allow re-selecting the same file
  }

  function onDrop(e) {
    e.preventDefault()
    dragDepth.current = 0
    setDrag(false)
    accept(e.dataTransfer.files?.[0])
  }

  async function analyzeSample(sampleCase) {
    setBusy(true)
    setError('')
    setFile(null)
    try {
      setResult(await api.mriAnalyzeSample(sampleCase))
    } catch (err) {
      setError(err.status ? err.message : 'The analysis service could not be reached — try again in a moment.')
    } finally {
      setBusy(false)
    }
  }

  async function analyze() {
    if (!file) return
    setBusy(true)
    setError('')
    // Note: the previous result stays visible until a new one succeeds.
    try {
      const fd = new FormData()
      fd.append('file', file, file.name)
      setResult(await api.mriAnalyze(fd))
    } catch (err) {
      const human = err.status
        ? err.message // server-side messages are already written for people
        : 'The analysis service could not be reached — the connection dropped or the server is waking up. Your file is still selected; try again in a moment.'
      setError(human)
    } finally {
      setBusy(false)
    }
  }

  if (info === null) {
    return (
      <div className="text-center" style={{ paddingTop: 60 }}>
        <div className="spinner" />
      </div>
    )
  }

  if (info.error) {
    return (
      <div className="card">
        <div className="empty">
          <div className="empty-icon">⚠</div>
          <h3>Couldn't reach the analysis service</h3>
          <p className="muted small mt-1">
            The server may be waking up — this can take up to a minute on the
            free hosting tier.
          </p>
          <button className="btn btn-primary mt-2" onClick={loadInfo}>
            Try again
          </button>
        </div>
      </div>
    )
  }

  if (!info.available) {
    return (
      <div className="card">
        <div className="empty">
          <div className="empty-icon">◉</div>
          <h3>MRI analysis is not available on this deployment</h3>
          <p className="muted small mt-1">
            This feature runs where the analysis model is installed — for
            example the local research setup. The rest of NeuroGuard works
            normally.
          </p>
        </div>
      </div>
    )
  }

  const impaired = result && result.predicted_index > 0
  const nClasses = result ? Object.keys(result.probabilities).length : 0

  return (
    <div className="grid" style={{ gap: 20 }}>
      <div className="card">
        <div className="card-title">Analyse a brain MRI (optional)</div>
        <div className="card-sub">
          Already have a brain scan from a checkup? Add a structural check on
          top of your cognitive tests. Sixteen axial slices are extracted and
          classified against OASIS-3. No scan? The cognitive tests are the
          heart of NeuroGuard — this step is entirely optional.
        </div>

        <div className="grid" style={{ gap: 12, maxWidth: 520 }}>
          <input
            ref={inputRef}
            type="file"
            accept=".nii,.nii.gz,.zip"
            onChange={pick}
            style={{ display: 'none' }}
          />

          {!file ? (
            <div
              className={`dropzone ${drag ? 'drag' : ''}`}
              role="button"
              tabIndex={0}
              aria-label="Choose an MRI scan file"
              onClick={() => inputRef.current?.click()}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  inputRef.current?.click()
                }
              }}
              onDragEnter={(e) => { e.preventDefault(); dragDepth.current += 1; setDrag(true) }}
              onDragOver={(e) => e.preventDefault()}
              onDragLeave={() => {
                dragDepth.current -= 1
                if (dragDepth.current <= 0) { dragDepth.current = 0; setDrag(false) }
              }}
              onDrop={onDrop}
            >
              <div className="dropzone-icon">⇪</div>
              <div className="dropzone-title">Tap or drop your MRI scan here</div>
              <div className="dropzone-hint">
                or <span style={{ color: 'var(--brand)', fontWeight: 600 }}>browse files</span> — a .nii/.nii.gz file, or your scan folder zipped (.zip), up to {MAX_MB} MB
              </div>
            </div>
          ) : (
            <div className="file-chip">
              <div className="file-chip-icon">NII</div>
              <div style={{ minWidth: 0 }}>
                <div className="file-chip-name">{file.name}</div>
                <div className="file-chip-meta">
                  {(file.size / (1024 * 1024)).toFixed(1)} MB · NIfTI volume
                </div>
              </div>
              <button
                className="file-chip-x"
                aria-label="Remove file"
                title="Remove file"
                onClick={() => { setFile(null); setError('') }}
              >
                ✕
              </button>
            </div>
          )}

          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <button className="btn btn-primary" disabled={!file || busy} onClick={analyze}>
              {busy ? 'Analysing…' : 'Analyse scan'}
            </button>
            {file && !busy && (
              <button className="btn btn-secondary" onClick={() => inputRef.current?.click()}>
                Choose a different file
              </button>
            )}
          </div>

          {info?.samples_available && (
            <div>
              <div className="tiny muted" style={{ marginBottom: 8 }}>
                No scan handy? Try it with an anonymized research scan:
              </div>
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                <button className="btn btn-secondary" disabled={busy} onClick={() => analyzeSample('normal')}>
                  Sample: healthy scan
                </button>
                <button className="btn btn-secondary" disabled={busy} onClick={() => analyzeSample('impaired')}>
                  Sample: impaired scan
                </button>
              </div>
            </div>
          )}

          <details className="explain-panel">
            <summary style={{ cursor: 'pointer', fontWeight: 650 }}>
              Where do I get my scan?
            </summary>
            <ol style={{ margin: '10px 0 0', paddingLeft: 18, lineHeight: 1.8 }}>
              <li>
                If you've ever had a brain MRI — even for headaches or a routine
                checkup — the scanning centre gave you a CD, pen drive or a
                download link. That data is yours; if you've lost it, the centre
                can re-issue a digital copy on request.
              </li>
              <li>
                Copy the scan folder from the CD to your computer, right-click
                it and compress it to a <strong>.zip</strong>, then drop the zip
                above. We read the hospital's format (DICOM) directly and pick
                the right series automatically.
              </li>
              <li>
                Never had an MRI? You don't need one. The six cognitive tests
                are the core of NeuroGuard — this page just adds an optional
                structural check for people who already have a scan.
              </li>
            </ol>
          </details>
        </div>

        <div aria-live="polite">
          {busy && (
            <div className="row mt-2" style={{ gap: 12 }}>
              <div className="spinner" style={{ width: 22, height: 22 }} />
              <div className="small muted">
                Uploading the scan and analysing 16 slices — hang tight, this can
                take a little while on a slow connection.
              </div>
            </div>
          )}
        </div>
        {error && <div className="alert alert-error mt-3" role="alert">{error}</div>}
      </div>

      {result && (
        <div className="card" ref={resultRef} tabIndex={-1}>
          <div className="card-title">Screening result</div>
          <div className="card-sub">
            Mean of {result.slices_analysed} slice predictions · volume{' '}
            {result.volume_shape?.join(' × ')}
          </div>

          <div className="mri-verdict">
            <div className={`verdict-text ${verdictClass(result.predicted_index, nClasses)}`}>
              {result.predicted_class}
            </div>
            <div className="verdict-sub">
              Model confidence {(result.confidence * 100).toFixed(1)}%{' '}
              <span className={`badge ${classStyle(result.predicted_index, nClasses)}`} style={{ marginLeft: 6 }}>
                screening indicator — not a diagnosis
              </span>
            </div>
          </div>

          {impaired && (
            <div className="alert alert-warn" role="note">{HONESTY_TEXT}</div>
          )}

          {Object.entries(result.probabilities).map(([name, p], i, all) => (
            <div key={name} className="prob-row">
              <div className="prob-head">
                <span>{name}</span>
                <span className="mono">{(p * 100).toFixed(1)}%</span>
              </div>
              <div className="prob-track">
                <div
                  className="prob-fill"
                  style={{ width: `${Math.max(2, p * 100)}%`, background: classColor(i, all.length) }}
                />
              </div>
            </div>
          ))}

          <PlainExplanation
            result={result}
            fileName={result.sample ? `the ${result.sample} sample scan` : file?.name}
          />
          {!impaired && (
            <p className="tiny muted mt-2" style={{ lineHeight: 1.6 }}>{HONESTY_TEXT}</p>
          )}

          <div className="disclaimer mt-3">{result.note}</div>
        </div>
      )}

      {result && <SliceViewer result={result} />}

      {info?.available && (
        <div className="grid grid-2">
          <div className="card">
            <div className="card-title">About this model</div>
            <div className="card-sub">{info.model}</div>
            <ul className="small muted" style={{ paddingLeft: 18, margin: 0, lineHeight: 1.9 }}>
              <li>{info.dataset?.name}</li>
              <li>
                {info.dataset?.subjects} subjects · {info.dataset?.sessions} MR sessions ·{' '}
                {info.dataset?.slices} slices
              </li>
              <li>Split {info.dataset?.split} — no subject appears in two splits</li>
              <li>{info.aggregation}</li>
            </ul>
            {info.cv ? (
              <div className="grid grid-3 mt-3" style={{ gap: 12 }}>
                <div>
                  <div className="stat-value">{(info.cv.accuracy.mean * 100).toFixed(0)}%</div>
                  <div className="stat-label">
                    accuracy ± {(info.cv.accuracy.std * 100).toFixed(0)} ({info.cv.folds}-fold CV)
                  </div>
                </div>
                <div>
                  <div className="stat-value">{info.cv.f1.mean.toFixed(2)}</div>
                  <div className="stat-label">F1 ± {info.cv.f1.std.toFixed(2)}</div>
                </div>
                <div>
                  <div className="stat-value">{info.cv.auc.mean.toFixed(2)}</div>
                  <div className="stat-label">AUC ± {info.cv.auc.std.toFixed(2)}</div>
                </div>
              </div>
            ) : info.test && (
              <div className="grid grid-3 mt-3" style={{ gap: 12 }}>
                <div>
                  <div className="stat-value">{(info.test.accuracy * 100).toFixed(0)}%</div>
                  <div className="stat-label">test accuracy</div>
                </div>
                <div>
                  <div className="stat-value">{info.test.macro_f1.toFixed(2)}</div>
                  <div className="stat-label">macro F1</div>
                </div>
                <div>
                  <div className="stat-value">{info.test.macro_auc_ovr.toFixed(2)}</div>
                  <div className="stat-label">macro AUC</div>
                </div>
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-title">Known limitations</div>
            <div className="card-sub">Stated up front, not buried</div>
            <ul className="small muted" style={{ paddingLeft: 18, margin: 0, lineHeight: 1.9 }}>
              {(info.caveats || []).map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      <div className="disclaimer">
        The MRI result and the behavioural risk score are reported separately on
        purpose: there is no dataset pairing the two, so combining them into one
        number would be a made-up weighting. Research and educational use only.
      </div>
    </div>
  )
}
