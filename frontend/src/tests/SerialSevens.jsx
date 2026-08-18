import { useRef, useState } from 'react'

/**
 * Test 4 — Serial 7s (MoCA attention item).
 *
 * Subtract 7 from 100, five times.
 *
 * Each prompt chains from the number the user actually gave, not from the ideal
 * sequence. That mirrors the MoCA rule: one arithmetic slip should cost one
 * mark, not wipe out every answer after it.
 */
export default function SerialSevens({ spec, onComplete }) {
  const start = spec?.start ?? 100
  const steps = spec?.steps ?? 5
  const subtract = spec?.subtract ?? 7

  const [started, setStarted] = useState(false)
  const [responses, setResponses] = useState([])
  const [current, setCurrent] = useState('')
  const startRef = useRef(null)
  const inputRef = useRef(null)

  const previous =
    responses.length === 0
      ? start
      : responses[responses.length - 1] ?? start - subtract * responses.length

  function begin() {
    startRef.current = Date.now()
    setStarted(true)
    setTimeout(() => inputRef.current?.focus(), 40)
  }

  function submitStep(e) {
    e.preventDefault()
    const value = current.trim() === '' ? null : Number(current)
    const next = [...responses, Number.isFinite(value) ? value : null]
    setResponses(next)
    setCurrent('')

    if (next.length >= steps) {
      onComplete({ responses: next, start }, Date.now() - startRef.current)
    } else {
      setTimeout(() => inputRef.current?.focus(), 40)
    }
  }

  function skip() {
    const next = [...responses, null]
    setResponses(next)
    setCurrent('')
    if (next.length >= steps) {
      onComplete({ responses: next, start }, Date.now() - startRef.current)
    }
  }

  return (
    <div className="card test-card">
      <div className="test-head">
        <div className="test-name">4 · Serial Sevens</div>
        <div className="test-instruction">
          Subtract {subtract} each time, starting from {start}. Do it in your
          head — no calculator, no paper.
        </div>
      </div>

      <div className="test-body">
        {!started ? (
          <div className="text-center">
            <p className="muted mb-3">
              You will be asked {steps} times. If you lose track, you can skip a
              step rather than guess.
            </p>
            <button className="btn btn-primary btn-lg" onClick={begin}>
              Start
            </button>
          </div>
        ) : (
          <form onSubmit={submitStep}>
            <div className="chain">
              <div className="chain-step">
                <span className="chain-from mono">{previous}</span>
                <span className="chain-from">−</span>
                <span className="chain-from mono">{subtract}</span>
                <span className="chain-from">=</span>
                <input
                  ref={inputRef}
                  className="num-input"
                  type="number"
                  value={current}
                  onChange={(e) => setCurrent(e.target.value)}
                  placeholder="?"
                  autoComplete="off"
                />
              </div>
            </div>

            <p className="stage-note mt-3">
              Step {responses.length + 1} of {steps}
            </p>

            <div className="row" style={{ justifyContent: 'center', marginTop: 18 }}>
              <button className="btn btn-secondary" type="button" onClick={skip}>
                Skip
              </button>
              <button className="btn btn-primary" type="submit" disabled={current === ''}>
                {responses.length + 1 >= steps ? 'Finish' : 'Next'}
              </button>
            </div>
          </form>
        )}
      </div>

      <div className="test-foot">
        <span className="muted small">
          {responses.length > 0
            ? `Answers so far: ${responses.map((r) => (r === null ? '—' : r)).join(', ')}`
            : 'Mental arithmetic only'}
        </span>
      </div>
    </div>
  )
}
