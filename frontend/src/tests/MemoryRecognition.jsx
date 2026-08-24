import { useEffect, useMemo, useState } from 'react'

/**
 * Test 1 — Delayed recognition memory.
 *
 * A set of objects is studied, then a filled delay, then a grid of those objects
 * mixed with twice as many distractors, from which the studied ones must be
 * picked out. How many objects, and how long the delay, are set by the age
 * difficulty tier — everything here reads from `spec`, so counts are never
 * assumed. See backend/app/ml/stimuli.py.
 *
 * The delay is *filled* with an unrelated counting task rather than left blank.
 * An empty pause lets the user silently rehearse the list, which turns a memory
 * test into a test of whether they thought to rehearse.
 */
export default function MemoryRecognition({ spec, onComplete }) {
  const [phase, setPhase] = useState('intro')
  const [remaining, setRemaining] = useState(0)
  const [selected, setSelected] = useState([])
  const [distractorCount, setDistractorCount] = useState(0)
  const [startedAt, setStartedAt] = useState(null)

  const studySeconds = spec?.study_seconds ?? 8
  const delaySeconds = spec?.delay_seconds ?? 10
  const targets = spec?.targets ?? []
  const grid = useMemo(() => spec?.grid ?? [], [spec])

  // Spelled out rather than shown as a digit — the instruction reads better, and
  // the count varies by difficulty tier (4 to 7), so it cannot be hardcoded.
  const countWord =
    ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'][
      targets.length
    ] ?? String(targets.length)

  // Drives the countdown for both the study and delay phases.
  useEffect(() => {
    if (phase !== 'study' && phase !== 'delay') return
    if (remaining <= 0) {
      if (phase === 'study') {
        setPhase('delay')
        setRemaining(delaySeconds)
      } else {
        setPhase('recognise')
      }
      return
    }
    const t = setTimeout(() => setRemaining((r) => r - 1), 1000)
    return () => clearTimeout(t)
  }, [phase, remaining, delaySeconds])

  function begin() {
    setStartedAt(Date.now())
    setRemaining(studySeconds)
    setPhase('study')
  }

  function toggle(item) {
    setSelected((prev) =>
      prev.includes(item) ? prev.filter((i) => i !== item) : [...prev, item]
    )
  }

  function finish() {
    onComplete({ selected }, startedAt ? Date.now() - startedAt : null)
  }

  return (
    <div className="card test-card">
      <div className="test-head">
        <div className="test-name">1 · Memory Recognition</div>
        <div className="test-instruction">
          {phase === 'intro' &&
            `You will see ${countWord} objects. Remember them.`}
          {phase === 'study' && `Memorise these ${countWord} objects.`}
          {phase === 'delay' && 'Hold them in mind while you do this.'}
          {phase === 'recognise' &&
            `Select the ${countWord} objects you saw. Choosing extras will count against you.`}
        </div>
      </div>

      <div className="test-body">
        {phase === 'intro' && (
          <div className="text-center">
            <p className="muted mb-3">
              The list appears for {studySeconds} seconds, then there is a short
              gap before you are asked to identify them.
            </p>
            <button className="btn btn-primary btn-lg" onClick={begin}>
              Start
            </button>
          </div>
        )}

        {phase === 'study' && (
          <>
            <div className="countdown mb-2">{remaining}</div>
            <div className="object-grid">
              {targets.map((item) => (
                <div key={item} className="object-tile study">
                  {item}
                </div>
              ))}
            </div>
          </>
        )}

        {phase === 'delay' && (
          <div className="text-center">
            <div className="countdown">{remaining}</div>
            <p className="stage-note mb-3">
              Tap the button each time — this stops you rehearsing the list.
            </p>
            <button
              className="btn btn-secondary btn-lg"
              onClick={() => setDistractorCount((c) => c + 1)}
            >
              Tap ({distractorCount})
            </button>
          </div>
        )}

        {phase === 'recognise' && (
          <>
            <div className="object-grid">
              {grid.map((item) => (
                <button
                  key={item}
                  type="button"
                  className={`object-tile ${selected.includes(item) ? 'selected' : ''}`}
                  onClick={() => toggle(item)}
                >
                  {item}
                </button>
              ))}
            </div>
            <p className="stage-note mt-3">
              {selected.length} of 5 selected
            </p>
          </>
        )}
      </div>

      <div className="test-foot">
        <span className="muted small">
          {phase === 'recognise'
            ? 'Choose exactly the ones you remember'
            : 'Do not write anything down'}
        </span>
        {phase === 'recognise' && (
          <button
            className="btn btn-primary"
            onClick={finish}
            disabled={selected.length === 0}
          >
            Continue
          </button>
        )}
      </div>
    </div>
  )
}
