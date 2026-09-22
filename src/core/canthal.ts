/**
 * canthal.ts — Geometric Head-Invariant Gaze Coordinate Normalization.
 *
 * CANCELS:
 * 1. Head translation in x, y via canthal midpoint offset subtraction.
 * 2. Head-to-camera distance variation via inter-canthal width normalization.
 * 3. Head roll angle via projection onto the eye's intrinsic basis vectors (u, v).
 */

export interface Point2D {
  x: number
  y: number
}

export interface EyeCornerLandmarks {
  inner: Point2D
  outer: Point2D
  irisCenter: Point2D
  irisRadius?: number
}

export interface CanthalProjection {
  gx: number
  gy: number
  width: number
  u: Point2D
  v: Point2D
}

/**
 * Projects iris coordinate onto the eye's intrinsic canthal coordinate system.
 *
 * @param eye Corner landmarks and iris center in pixel coordinates
 * @param isLeftEye True for anatomical left eye, false for right (adjusts outer/inner orientation)
 */
export function projectCanthalGaze(
  eye: EyeCornerLandmarks,
  isLeftEye: boolean
): CanthalProjection {
  const { inner, outer, irisCenter } = eye

  // 1. Inter-canthal width (px)
  const dx = outer.x - inner.x
  const dy = outer.y - inner.y
  const width = Math.hypot(dx, dy) || 1.0

  // 2. Canthal midpoint (moves with head translation)
  const mx = (inner.x + outer.x) / 2.0
  const my = (inner.y + outer.y) / 2.0

  // 3. Intrinsic horizontal axis u (unit vector from inner to outer corner)
  // Left eye outer is image-left (needs sign inversion), right eye outer is image-right.
  const sign = isLeftEye ? -1.0 : 1.0
  const ux = (sign * dx) / width
  const uy = (sign * dy) / width

  // 4. Intrinsic vertical axis v (orthonormal 90-degree rotation of u)
  const vx = -uy
  const vy = ux

  // 5. Iris displacement relative to canthal midpoint
  const rx = irisCenter.x - mx
  const ry = irisCenter.y - my

  // 6. Projected, dimensionless coordinates in units of eye-widths
  const gx = (rx * ux + ry * uy) / width
  const gy = (rx * vx + ry * vy) / width

  return {
    gx,
    gy,
    width,
    u: { x: ux, y: uy },
    v: { x: vx, y: vy },
  }
}
