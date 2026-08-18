import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import DigitSpanBackward from '../tests/DigitSpanBackward.jsx'
import MemoryRecognition from '../tests/MemoryRecognition.jsx'
import PictureDescription from '../tests/PictureDescription.jsx'
import SerialSevens from '../tests/SerialSevens.jsx'
import TrailMakingB from '../tests/TrailMakingB.jsx'
import VerbalFluency from '../tests/VerbalFluency.jsx'

// Order matters: the memory test runs first because its delayed-recognition
// phase is what the rest of the battery is implicitly filling time for.
const SEQUENCE = [
  { key: 'memory_recognition', Component: MemoryRecognition },
  { key: 'picture_description', Component: PictureDescription },
  { key: 'trail_making_b', Component: TrailMakingB },
  { key: 'serial_sevens', Component: SerialSevens },
  { key: 'digit_span_backward', Component: DigitSpanBackward },
  { key: 'verbal_fluency', Component: VerbalFluency },
]

export default function Assessment() {
  const navigate = useNavigate()
  const [session, setSession] = useState(null)
  const [stimuli, setStimuli] = useState(null)
  const [step, setStep] = useState(0)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [finishing, setFinishing] = useState(false)

  useEffect(() => {
    let cancelled = false
    api
      .startAssessment()
      .then((res) => {
        if (cancelled) return
        setSession(res.session)
        setStimuli(res.stimuli)
      })
      .catch((err) => !cancelled && setError(err.message || 'Could not start the assessment'))
    return () => {
      cancelled = true
    }
  }, [])

  const finish = useCallback(
    async (sessionId) => {
      setFinishing(true)
      try {
        await api.completeAssessment(sessionId)
        navigate(`/results/${sessionId}`, { replace: true })
      } catch (err) {
        setError(err.message || 'Could not finalise the assessment')
        setFinishing(false)
      }
    },
    [navigate]
  )

  const advance = useCallback(
    (sessionId) => {
      setStep((prev) => {
        const next = prev + 1
        if (next >= SEQUENCE.length) finish(sessionId)
        return next
      })
    },
    [finish]
  )

  // Structured tests hand back a raw response; this submits it and moves on.
  const handleComplete = useCallback(
    async (testName, rawResponse, durationMs) => {
      if (!session) return
      setBusy(true)
      setError('')
      try {
        await api.submitTest(session.id, {
          test_name: testName,
          raw_response: rawResponse,
          duration_ms: durationMs ?? null,
        })
        advance(session.id)
      } catch (err) {
        setError(err.message || 'Could not save that answer')
      } finally {
        setBusy(false)
      }
    },
    [session, advance]
  )

  // The picture test submits itself (audio upload or typed text), so it only
  // needs to tell us to move on.
  const handleLanguageComplete = useCallback(() => {
    if (session) advance(session.id)
  }, [session, advance])

  if (error && !session) {
    return (
      <div className="assess-wrap">
        <div className="alert alert-error">{error}</div>
        <button className="btn btn-secondary" onClick={() => window.location.reload()}>
          Try again
        </button>
      </div>
    )
  }

  if (!session || !stimuli) {
    return (
      <div className="assess-wrap text-center" style={{ paddingTop: 80 }}>
        <div className="spinner" />
        <p className="muted small mt-2">Preparing your assessment…</p>
      </div>
    )
  }

  if (finishing) {
    return (
      <div className="assess-wrap text-center" style={{ paddingTop: 80 }}>
        <div className="spinner" />
        <p className="muted mt-2">Analysing your results…</p>
      </div>
    )
  }

  const current = SEQUENCE[Math.min(step, SEQUENCE.length - 1)]
  const { Component, key } = current
  const progress = (step / SEQUENCE.length) * 100

  return (
    <div className="assess-wrap">
      <div className="progress-row">
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        <span className="progress-text">
          {step + 1} / {SEQUENCE.length}
        </span>
      </div>
      <p className="muted small mb-3">
        Work somewhere quiet, and answer as accurately as you can. You cannot go
        back to a previous test.
      </p>

      {error && <div className="alert alert-error">{error}</div>}
      {busy && <div className="alert alert-info">Saving…</div>}

      {key === 'picture_description' ? (
        <PictureDescription
          key={key}
          spec={stimuli[key]}
          sessionId={session.id}
          api={api}
          onComplete={handleLanguageComplete}
          onError={(err) => setError(err.message)}
        />
      ) : (
        <Component
          key={key}
          spec={stimuli[key]}
          onComplete={(raw, ms) => handleComplete(key, raw, ms)}
        />
      )}
    </div>
  )
}
