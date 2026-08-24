const BAND_COLOUR = {
  Low: 'var(--low)',
  Borderline: 'var(--borderline)',
  Elevated: 'var(--elevated)',
}

/**
 * Headline score dial.
 *
 * Shows a score out of 10 where HIGHER IS BETTER, which is how people read a
 * score without being told. The underlying risk runs the other way, so the arc
 * fills with the score, not the risk: a full green ring means a good result.
 *
 * `band` still comes from the risk, so the colour and the number always agree —
 * a high score is green, a low one red.
 */
export default function RiskGauge({ score = 0, band = 'Low', size = 168 }) {
  const stroke = 13
  const radius = (size - stroke) / 2
  const circumference = 2 * Math.PI * radius
  const clamped = Math.max(0, Math.min(10, Number(score) || 0))
  const colour = BAND_COLOUR[band] || 'var(--brand)'

  return (
    <div className="gauge" style={{ width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke="var(--surface-2)" strokeWidth={stroke}
        />
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke={colour} strokeWidth={stroke} strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - clamped / 10)}
          style={{ transition: 'stroke-dashoffset .8s ease' }}
        />
      </svg>
      <div className="gauge-label">
        <div className="gauge-value" style={{ color: colour }}>
          {clamped.toFixed(1)}
          <span className="gauge-outof">/10</span>
        </div>
        <div className="gauge-unit">cognitive score</div>
      </div>
    </div>
  )
}
