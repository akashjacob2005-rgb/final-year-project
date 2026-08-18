import { useMemo, useRef, useState } from 'react'

/**
 * Test 3 — Trail Making B (MoCA alternation task).
 *
 * Click 1 → A → 2 → B → 3 → C → 4 → D → 5 → E. Measures the executive cost of
 * switching between two sequences.
 *
 * Every click is recorded, including wrong ones. The server decides what counts
 * as correct from the stored stimuli — the client never has the expected path,
 * so it cannot leak the answer or mark its own work.
 */
export default function TrailMakingB({ spec, onComplete }) {
  const nodes = useMemo(() => spec?.nodes ?? [], [spec])
  const [started, setStarted] = useState(false)
  const [clicked, setClicked] = useState([])
  const [done, setDone] = useState([])
  const [errorNode, setErrorNode] = useState(null)
  const startRef = useRef(null)

  // Reconstruct the expected order locally *only* to give immediate visual
  // feedback. Scoring is server-side regardless of what happens here.
  const expected = useMemo(() => {
    const nums = nodes.map((n) => n.label).filter((l) => /^\d$/.test(l)).sort()
    const lets = nodes.map((n) => n.label).filter((l) => /^[A-Z]$/.test(l)).sort()
    return nums.flatMap((n, i) => [n, lets[i]]).filter(Boolean)
  }, [nodes])

  function begin() {
    startRef.current = Date.now()
    setStarted(true)
  }

  function handleClick(label) {
    if (!started || done.includes(label)) return

    const next = [...clicked, label]
    setClicked(next)

    if (label === expected[done.length]) {
      const advanced = [...done, label]
      setDone(advanced)
      if (advanced.length === expected.length) {
        onComplete(
          { clicked: next, duration_ms: Date.now() - startRef.current },
          Date.now() - startRef.current
        )
      }
    } else {
      setErrorNode(label)
      setTimeout(() => setErrorNode(null), 340)
    }
  }

  const nextLabel = expected[done.length]

  return (
    <div className="card test-card">
      <div className="test-head">
        <div className="test-name">3 · Trail Making B</div>
        <div className="test-instruction">
          Click the circles alternating number, letter, number, letter — 1, A, 2,
          B, and so on, as quickly as you can.
        </div>
      </div>

      <div className="test-body">
        {!started ? (
          <div className="text-center">
            <p className="muted mb-3">
              Start at <strong>1</strong>, then <strong>A</strong>, then{' '}
              <strong>2</strong>, then <strong>B</strong> … up to{' '}
              <strong>E</strong>. Both speed and accuracy are recorded.
            </p>
            <button className="btn btn-primary btn-lg" onClick={begin}>
              Start
            </button>
          </div>
        ) : (
          <>
            <div className="trail-canvas">
              <svg className="trail-line" viewBox="0 0 100 100" preserveAspectRatio="none">
                {done.slice(1).map((label, i) => {
                  const a = nodes.find((n) => n.label === done[i])
                  const b = nodes.find((n) => n.label === label)
                  if (!a || !b) return null
                  return (
                    <line
                      key={label}
                      x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                      stroke="var(--low)" strokeWidth="0.5" vectorEffect="non-scaling-stroke"
                    />
                  )
                })}
              </svg>

              {nodes.map((node) => (
                <button
                  key={node.label}
                  type="button"
                  className={`trail-node ${done.includes(node.label) ? 'done' : ''} ${
                    errorNode === node.label ? 'error' : ''
                  }`}
                  style={{ left: `${node.x}%`, top: `${node.y}%` }}
                  onClick={() => handleClick(node.label)}
                  disabled={done.includes(node.label)}
                >
                  {node.label}
                </button>
              ))}
            </div>
            <p className="stage-note mt-2">
              Next: <strong>{nextLabel}</strong> · {done.length} of{' '}
              {expected.length} connected
            </p>
          </>
        )}
      </div>

      <div className="test-foot">
        <span className="muted small">
          {started ? 'A wrong circle shakes and is recorded as an error' : 'Speed matters'}
        </span>
      </div>
    </div>
  )
}
