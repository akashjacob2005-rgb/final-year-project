const BAND_COLOUR = {
  Low: 'var(--low)',
  Borderline: 'var(--borderline)',
  Elevated: 'var(--elevated)',
}

export default function RiskGauge({ percent = 0, band = 'Low', size = 168 }) {
  const stroke = 13
  const radius = (size - stroke) / 2
  const circumference = 2 * Math.PI * radius
  const clamped = Math.max(0, Math.min(100, percent))
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
          strokeDashoffset={circumference * (1 - clamped / 100)}
          style={{ transition: 'stroke-dashoffset .8s ease' }}
        />
      </svg>
      <div className="gauge-label">
        <div className="gauge-value" style={{ color: colour }}>
          {clamped.toFixed(0)}%
        </div>
        <div className="gauge-unit">screening score</div>
      </div>
    </div>
  )
}
