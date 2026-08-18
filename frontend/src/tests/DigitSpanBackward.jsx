import { useEffect, useRef, useState } from 'react'

/**
 * Test 5 — Digit span backward (working memory).
 *
 * Digits are shown one at a time, then must be typed in reverse order. Trials
 * get longer (3, 4, 5, 6 digits).
 *
 * Digits are presented sequentially rather than all at once so the task loads
 * working memory instead of visual scanning — the whole string on screen could
 * simply be read backwards.
 */
export default function DigitSpanBackward({ spec, onComplete }) {
  const trials = spec?.trials ?? []
  const intervalMs = spec?.digit_interval_ms ?? 1000

  const [trialIndex, setTrialIndex] = useState(0)
  const [phase, setPhase] = useState('intro') // intro | show | recall | done
  const [digitIndex, setDigitIndex] = useState(0)
  const [answer, setAnswer] = useState('')
  const [results, setResults] = useState([])
  const startRef = useRef(null)
  const inputRef = useRef(null)

  const trial = trials[trialIndex]

  // Step through the digits of the current trial.
  useEffect(() => {
    if (phase !== 'show' || !trial) return

    if (digitIndex >= trial.sequence.length) {
      const t = setTimeout(() => {
        setPhase('recall')
        setTimeout(() => inputRef.current?.focus(), 40)
      }, 400)
      return () => clearTimeout(t)
    }

    const t = setTimeout(() => setDigitIndex((i) => i + 1), intervalMs)
    return () => clearTimeout(t)
  }, [phase, digitIndex, trial, intervalMs])

  function beginTrial() {
    if (!startRef.current) startRef.current = Date.now()
    setDigitIndex(0)
    setAnswer('')
    setPhase('show')
  }

  function submitTrial(e) {
    e?.preventDefault()
    const response = answer.replace(/\D/g, '').split('').map(Number)
    const next = [...results, { response }]
    setResults(next)

    if (trialIndex + 1 >= trials.length) {
      setPhase('done')
      onComplete({ trials: next }, Date.now() - startRef.current)
    } else {
      setTrialIndex((i) => i + 1)
      setPhase('ready')
      setAnswer('')
    }
  }

  const showingDigit =
    phase === 'show' && trial && digitIndex < trial.sequence.length
      ? trial.sequence[digitIndex]
      : null

  return (
    <div className="card test-card">
      <div className="test-head">
        <div className="test-name">5 · Digit Span Backward</div>
        <div className="test-instruction">
          Watch the digits appear one by one, then type them in{' '}
          <strong>reverse</strong> order.
        </div>
      </div>

      <div className="test-body">
        {phase === 'intro' && (
          <div className="text-center">
            <p className="muted mb-3">
              For example, if you see <strong>2 — 7 — 4</strong>, you type{' '}
              <strong>472</strong>. There are {trials.length} rounds, each
              longer than the last.
            </p>
            <button className="btn btn-primary btn-lg" onClick={beginTrial}>
              Start
            </button>
          </div>
        )}

        {phase === 'ready' && (
          <div className="text-center">
            <p className="muted mb-3">
              Round {trialIndex + 1} of {trials.length} —{' '}
              {trial?.sequence.length} digits.
            </p>
            <button className="btn btn-primary btn-lg" onClick={beginTrial}>
              Show digits
            </button>
          </div>
        )}

        {phase === 'show' && (
          <div className="text-center">
            <div className="digit-display">{showingDigit ?? ''}</div>
            <p className="stage-note">Watch carefully…</p>
          </div>
        )}

        {phase === 'recall' && (
          <form className="text-center" onSubmit={submitTrial}>
            <p className="muted mb-2">Type the digits in reverse order</p>
            <input
              ref={inputRef}
              className="num-input"
              style={{ width: 180, fontSize: '1.5rem', letterSpacing: '.15em' }}
              value={answer}
              onChange={(e) => setAnswer(e.target.value.replace(/\D/g, ''))}
              inputMode="numeric"
              autoComplete="off"
              maxLength={8}
            />
            <div className="mt-3">
              <button className="btn btn-primary" type="submit">
                {trialIndex + 1 >= trials.length ? 'Finish' : 'Next round'}
              </button>
            </div>
          </form>
        )}

        {phase === 'done' && (
          <p className="stage-note">Recorded. Moving on…</p>
        )}
      </div>

      <div className="test-foot">
        <span className="muted small">
          Round {Math.min(trialIndex + 1, trials.length)} of {trials.length}
        </span>
      </div>
    </div>
  )
}
