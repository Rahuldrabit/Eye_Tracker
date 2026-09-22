/**
 * features28.ts — Physics-Informed 28-Term Gaze & Head Feature Extraction.
 *
 * Terms:
 * 1. Perspective foreshortening correction on iris radius.
 * 2. Biologically invariant optical metric depth (Z_ratio).
 * 3. Binocular iris 3D yaw disparity (delta_iris).
 * 4. Vertical and horizontal parallax coupling (head_y * depth, head_x * depth).
 * 5. Asymmetric independent eye weights (Eye A vs Eye B).
 * 6. Eyelid occlusion compensation (gy * ear).
 * 7. Screen tangent perspective cubic terms (gx^3, gy^3).
 * 8. Geometric gaze-head decoupling (gx * yaw, gy * pitch).
 */

export interface EyeLandmarks {
  gx: number
  gy: number
  ear: number
  width: number
  irisRadius: number
}

export interface EyeFeature {
  gx: number
  gy: number
  disparity: number
  ear: number
  faceScale: number
  yawProxy: number
  pitchProxy: number
  roll: number
  noseX: number
  noseY: number
  eyeA: EyeLandmarks
  eyeB: EyeLandmarks
}

export interface CalibrationAnchor {
  faceScale: number
  noseX: number
  noseY: number
  pitchProxy: number
  yawProxy: number
  irisRadius: number
}

export const ENHANCED_FEATURE_NAMES = [
  'intercept',
  'gx',
  'gy',
  'eyeA_gx',
  'eyeA_gy',
  'eyeB_gx',
  'eyeB_gy',
  'yaw',
  'pitch',
  'head_x',
  'head_y',
  'z_ratio',
  'iris_yaw_proxy',
  'head_y_parallax',
  'head_x_parallax',
  'pitch_parallax',
  'yaw_parallax',
  'gx_depth',
  'gy_depth',
  'gaze_head_x',
  'gaze_head_y',
  'gy_ear',
  'ear',
  'disparity',
  'vertical_disparity',
  'gx_cubed',
  'gy_cubed',
  'roll',
] as const

export const ENHANCED_FEATURE_COUNT = ENHANCED_FEATURE_NAMES.length // 28 terms

/**
 * Computes median reference anchor values across calibration samples.
 */
export function computeCalibrationAnchor(samples: { feature: EyeFeature }[]): CalibrationAnchor {
  const med = (fn: (f: EyeFeature) => number): number => {
    const vals = samples.map((s) => fn(s.feature)).sort((a, b) => a - b)
    if (vals.length === 0) return 0
    const mid = vals.length >> 1
    return vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2.0
  }

  const faceScale = med((f) => f.faceScale) || 1.0
  const noseX = med((f) => f.noseX)
  const noseY = med((f) => f.noseY)
  const pitchProxy = med((f) => f.pitchProxy)
  const yawProxy = med((f) => f.yawProxy)
  const irisRadius = med((f) => (f.eyeA.irisRadius + f.eyeB.irisRadius) / 2.0) || 10.0

  return { faceScale, noseX, noseY, pitchProxy, yawProxy, irisRadius }
}

/**
 * Builds the 28-term feature vector for a given EyeFeature using the calibration anchor.
 */
export function buildEnhancedFeatureRow(
  f: EyeFeature,
  anchor: CalibrationAnchor
): Float64Array {
  const row = new Float64Array(ENHANCED_FEATURE_COUNT)

  // Intercept
  row[0] = 1.0

  // 1. Core head-invariant gaze offsets
  row[1] = f.gx
  row[2] = f.gy

  // 2. Asymmetric independent eye offsets
  row[3] = f.eyeA.gx
  row[4] = f.eyeA.gy
  row[5] = f.eyeB.gx
  row[6] = f.eyeB.gy

  // 3. Normalized head-pose relative to anchor
  const yaw = f.yawProxy - anchor.yawProxy
  const pitch = f.pitchProxy - anchor.pitchProxy
  row[7] = yaw
  row[8] = pitch

  // 4. Normalized head position in screen space
  const headX = (f.noseX - anchor.noseX) / (anchor.faceScale || 1.0)
  const headY = (f.noseY - anchor.noseY) / (anchor.faceScale || 1.0)
  row[9] = headX
  row[10] = headY

  // 5. Optical depth proxy (Z_ratio)
  const currentIrisR = (f.eyeA.irisRadius + f.eyeB.irisRadius) / 2.0
  const zRatio = (f.faceScale / (anchor.faceScale || 1.0) + currentIrisR / (anchor.irisRadius || 1.0)) / 2.0
  row[11] = zRatio - 1.0

  // 6. Iris yaw disparity
  const irisYawProxy = (f.eyeA.irisRadius - f.eyeB.irisRadius) / (currentIrisR || 1.0)
  row[12] = irisYawProxy

  // 7. Parallax coupling
  row[13] = headY * row[11]
  row[14] = headX * row[11]
  row[15] = pitch * row[11]
  row[16] = yaw * row[11]

  // 8. Depth-scaled gaze offsets
  row[17] = f.gx * row[11]
  row[18] = f.gy * row[11]

  // 9. Gaze-head interaction terms
  row[19] = f.gx * yaw
  row[20] = f.gy * pitch

  // 10. Eyelid occlusion interaction (vertical gaze bias compensation)
  row[21] = f.gy * f.ear
  row[22] = f.ear

  // 11. Disparity terms
  row[23] = f.disparity
  row[24] = f.eyeA.gy - f.eyeB.gy

  // 12. Non-linear cubic screen-tangent projections
  row[25] = f.gx * f.gx * f.gx
  row[26] = f.gy * f.gy * f.gy

  // 13. Roll angle
  row[27] = f.roll

  return row
}
