import { useEffect, useRef, useState } from 'react'

/**
 * Test 6 — Letter (phonemic) verbal fluency.
 *
 * Name as many words as possible starting with the given letter in 60 seconds.
 * MoCA scores 11+ words as a pass.
 *
 * Duplicates are flagged in the UI but still submitted. The server does the
 * real filtering, and *how many* duplicates a person produces is itself
 * informative — repeating without noticing is a decline signal.
 */
export default function VerbalFluency({ spec, onComplete }) {
  const letter = (spec?.letter ?? 'F').toUpperCase()
  const duration = spec?.duration_seconds ?? 60

  const [started, setStarted] = useState(false)
  const [remaining, setRemaining] = useState(duration)
  const [words, setWords] = useState([])
  const [current, setCurrent] = useState('')
  const inputRef = useRef(null)
  const startRef = useRef(null)
  const finishedRef = useRef(false)

  useEffect(() => {
    if (!started || remaining <= 0) return
    const t = setTimeout(() => setRemaining((r) => r - 1), 1000)
    return () => clearTimeout(t)
  }, [started, remaining])

  // Auto-submit when the clock runs out. The guard stops a double submit if the
  // user also presses Finish at the same moment.
  useEffect(() => {
    if (started && remaining <= 0 && !finishedRef.current) {
      finishedRef.current = true
      onComplete({ words, letter }, Date.now() - startRef.current)
    }
  }, [started, remaining, words, letter, onComplete])

  function begin() {
    startRef.current = Date.now()
    setStarted(true)
    setTimeout(() => inputRef.current?.focus(), 40)
  }

  function addWord(e) {
    e.preventDefault()
    const w = current.trim().toLowerCase()
    if (w) setWords((prev) => [...prev, w])
    setCurrent('')
  }

  function finishNow() {
    if (finishedRef.current) return
    finishedRef.current = true
    onComplete({ words, letter }, Date.now() - startRef.current)
  }

  const seen = new Set()
  const marked = words.map((w) => {
    const dupe = seen.has(w)
    seen.add(w)
    return { word: w, dupe }
  })
  const uniqueValid = [...seen].filter((w) => w.startsWith(letter.toLowerCase())).length

  return (
    <div className="card test-card">
      <div className="test-head">
        <div className="test-name">6 · Verbal Fluency</div>
        <div className="test-instruction">
          Type as many words as you can beginning with the letter{' '}
          <strong>{letter}</strong>. No names of people or places, and no
          repeats.
        </div>
      </div>

      <div className="test-body">
        {!started ? (
          <div className="text-center">
            <p className="muted mb-3">
              You have {duration} seconds. Press Enter after each word.
            </p>
            <button className="btn btn-primary btn-lg" onClick={begin}>
              Start
            </button>
          </div>
        ) : (
          <>
            <div className="row between mb-2">
              <span className="muted small">
                {uniqueValid} valid word{uniqueValid === 1 ? '' : 's'}
              </span>
              <span
                className="rec-timer"
                style={{ color: remaining <= 10 ? 'var(--elevated)' : 'var(--brand)' }}
              >
                {String(Math.floor(remaining / 60)).padStart(2, '0')}:
                {String(remaining % 60).padStart(2, '0')}
              </span>
            </div>

            <form onSubmit={addWord}>
              <input
                ref={inputRef}
                className="input"
                value={current}
                onChange={(e) => setCurrent(e.target.value)}
                placeholder={`A word starting with ${letter}…`}
                autoComplete="off"
                disabled={remaining <= 0}
              />
            </form>

            <div className="word-chips">
              {marked.map((m, i) => (
                <span key={`${m.word}-${i}`} className={`word-chip ${m.dupe ? 'dupe' : ''}`}>
                  {m.word}
                  {m.dupe && ' ·repeat'}
                </span>
              ))}
            </div>
          </>
        )}
      </div>

      <div className="test-foot">
        <span className="muted small">
          {started ? 'Press Enter after each word' : `Letter: ${letter}`}
        </span>
        {started && remaining > 0 && (
          <button className="btn btn-secondary" onClick={finishNow}>
            I'm finished
          </button>
        )}
      </div>
    </div>
  )
}
