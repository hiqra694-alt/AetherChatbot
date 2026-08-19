'use client'

import { useId } from 'react'

interface AetherLogoProps {
  /** Pixel size (applied as inline width/height so it's exact regardless of
   * Tailwind config). Omit to size purely via `className` instead (e.g.
   * `className="w-6 h-6"` for a navbar/chat-bubble mark, `className="w-20
   * h-20"` for the welcome screen) -- falls back to a 40px default when
   * neither is given. */
  size?: number
  className?: string
}

// Aether's mascot mark: a glowing ethereal orb with a soft purple/pink/blue
// mesh wash fading into warm peach/amber at the bottom, plus two friendly
// upward-pointing "^ ^" caret eyes. Every instance needs its own
// gradient/filter ids -- several of these render on screen at once (sidebar
// header, every AI chat bubble, the welcome hero), and duplicate SVG ids
// would make later instances silently reuse the first one's <defs> --
// useId() keeps each instance's <defs> uniquely addressable. The colon
// useId() returns isn't valid inside a url(#...) reference, hence the strip.
export default function AetherLogo({ size, className = '' }: AetherLogoProps) {
  const uid = useId().replace(/:/g, '')
  const baseGradientId = `aether-base-${uid}`
  const pinkGradientId = `aether-pink-${uid}`
  const blueGradientId = `aether-blue-${uid}`
  const eyeGlowId = `aether-eye-glow-${uid}`

  return (
    <svg
      viewBox="0 0 100 100"
      role="img"
      aria-label="Aether"
      className={`${size ? '' : 'w-10 h-10'} ${className} flex-shrink-0`}
      style={{
        ...(size ? { width: size, height: size } : undefined),
        // Soft ambient glow around the orb, echoing its own gradient
        // (purple + peach) rather than a flat/neutral shadow.
        filter: 'drop-shadow(0 0 8px rgba(168,85,247,0.45)) drop-shadow(0 0 16px rgba(251,146,60,0.22))',
      }}
    >
      <defs>
        {/* Base sphere: a light lavender highlight up top easing through
            purple and a soft rose into a gentle peach base -- close,
            low-contrast stops (rather than few widely-spaced ones) keep the
            wash smooth instead of banding into a harsh, saturated patch. */}
        <linearGradient id={baseGradientId} x1="50%" y1="0%" x2="50%" y2="100%">
          <stop offset="0%" stopColor="#D8B4FE" />
          <stop offset="30%" stopColor="#A855F7" />
          <stop offset="55%" stopColor="#F0ABFC" />
          <stop offset="78%" stopColor="#FCA5A5" />
          <stop offset="100%" stopColor="#FDBA74" />
        </linearGradient>
        {/* Two overlapping radial blobs give the base wash its "mesh
            gradient" quality -- pink upper-left, blue upper-right. A middle
            stop on each (rather than jumping straight to transparent) lets
            them feather out softly instead of leaving a visible hard edge. */}
        <radialGradient id={pinkGradientId} cx="30%" cy="24%" r="65%">
          <stop offset="0%" stopColor="#F9A8D4" stopOpacity="0.55" />
          <stop offset="55%" stopColor="#F9A8D4" stopOpacity="0.18" />
          <stop offset="100%" stopColor="#F9A8D4" stopOpacity="0" />
        </radialGradient>
        <radialGradient id={blueGradientId} cx="74%" cy="26%" r="65%">
          <stop offset="0%" stopColor="#93C5FD" stopOpacity="0.5" />
          <stop offset="55%" stopColor="#93C5FD" stopOpacity="0.16" />
          <stop offset="100%" stopColor="#93C5FD" stopOpacity="0" />
        </radialGradient>
        <filter id={eyeGlowId} x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="1.4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Outer Shape: smooth rounded circle carrying the mesh gradient. */}
      <circle cx="50" cy="50" r="46" fill={`url(#${baseGradientId})`} />
      <circle cx="50" cy="50" r="46" fill={`url(#${pinkGradientId})`} />
      <circle cx="50" cy="50" r="46" fill={`url(#${blueGradientId})`} />

      {/* Character Eyes: friendly upward-pointing "^ ^" carets, clustered in
          the upper-left/center of the orb and tilted slightly toward each
          other. Each is an open 3-point path (base -> peak -> base) rather
          than a filled shape -- the thick stroke with round linecap/linejoin
          is what makes every corner, including the peak, unavoidably soft. */}
      <g
        filter={`url(#${eyeGlowId})`}
        fill="none"
        stroke="white"
        strokeWidth={5.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M30 45 L36 35 L42 45" transform="rotate(10 36 40)" />
        <path d="M46 45 L52 35 L58 45" transform="rotate(-10 52 40)" />
      </g>
    </svg>
  )
}
