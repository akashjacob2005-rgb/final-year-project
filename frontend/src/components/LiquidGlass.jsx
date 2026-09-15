/**
 * Liquid-glass primitives, ported from the 21st.dev "liquid glass" component
 * (shadcn/Tailwind/TS original) to this project's stack: plain JSX + the
 * design-system classes in index.css. The glass anatomy is identical —
 * distortion layer (SVG filter + backdrop blur), white tint, inset shine,
 * content on top — and the bouncy cubic-bezier(0.175, 0.885, 0.32, 2.2)
 * easing is preserved in CSS. The demo dock/marketing chrome was dropped;
 * only the reusable pieces live here.
 *
 * Usage: mount <GlassFilter /> once on any page that renders a <GlassPanel>.
 */

export function GlassFilter() {
  return (
    <svg style={{ display: 'none' }} aria-hidden="true">
      <filter
        id="glass-distortion"
        x="0%"
        y="0%"
        width="100%"
        height="100%"
        filterUnits="objectBoundingBox"
      >
        <feTurbulence
          type="fractalNoise"
          baseFrequency="0.001 0.005"
          numOctaves="1"
          seed="17"
          result="turbulence"
        />
        <feComponentTransfer in="turbulence" result="mapped">
          <feFuncR type="gamma" amplitude="1" exponent="10" offset="0.5" />
          <feFuncG type="gamma" amplitude="0" exponent="1" offset="0" />
          <feFuncB type="gamma" amplitude="0" exponent="1" offset="0.5" />
        </feComponentTransfer>
        <feGaussianBlur in="turbulence" stdDeviation="3" result="softMap" />
        <feSpecularLighting
          in="softMap"
          surfaceScale="5"
          specularConstant="1"
          specularExponent="100"
          lightingColor="white"
          result="specLight"
        >
          <fePointLight x="-200" y="-200" z="300" />
        </feSpecularLighting>
        <feComposite
          in="specLight"
          operator="arithmetic"
          k1="0"
          k2="1"
          k3="1"
          k4="0"
          result="litImage"
        />
        <feDisplacementMap
          in="SourceGraphic"
          in2="softMap"
          scale="200"
          xChannelSelector="R"
          yChannelSelector="G"
        />
      </filter>
    </svg>
  )
}

export function GlassPanel({ children, className = '', style = {} }) {
  return (
    <div className={`liquid-glass ${className}`} style={style}>
      <div className="lg-distort" />
      <div className="lg-tint" />
      <div className="lg-shine" />
      <div className="lg-content">{children}</div>
    </div>
  )
}
