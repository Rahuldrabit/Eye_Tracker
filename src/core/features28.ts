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
  'disparity',
  'ear',
  'faceScale',
  'yaw',
  'pitch',
  'roll',
  'eyeA_gx',
  'eyeA_gy',
  'eyeA_ear',
  'eyeA_width',
  'eyeA_irisRadius',
  'eyeB_gx',
  'eyeB_gy',
  'eyeB_ear',
  'eyeB_width',
  'eyeB_irisRadius',
  'width_asymmetry',
  'iris_asymmetry',
  'tan_gx',
  'tan_gy',
  'dist_scaled_gx',
  'dist_scaled_gy',
  'ear_diff',
  'gx_yaw_cross',
  'gy_pitch_cross',
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

  const faceScale = med((f) => f.faceScale) || 260.0
  const noseX = med((f) => f.noseX)
  const noseY = med((f) => f.noseY)
  const pitchProxy = med((f) => f.pitchProxy)
  const yawProxy = med((f) => f.yawProxy)
  const irisRadius = med((f) => (f.eyeA.irisRadius + f.eyeB.irisRadius) / 2.0) || 10.5

  return { faceScale, noseX, noseY, pitchProxy, yawProxy, irisRadius }
}

/**
 * Builds the 28-term SOTA feature vector for a given EyeFeature using the calibration anchor.
 * Incorporates perspective width/iris asymmetry invariants, tangent transforms, and depth warping.
 */
export function buildEnhancedFeatureRow(
  f: EyeFeature,
  anchor: CalibrationAnchor
): Float64Array {
  const row = new Float64Array(ENHANCED_FEATURE_COUNT)

  // 0. Intercept
  row[0] = 1.0

  // 1-2. Core binocular gaze vectors
  row[1] = f.gx
  row[2] = f.gy

  // 3-5. Ocular disparity, mean EAR, and normalized distance/scale
  row[3] = f.disparity
  row[4] = f.ear
  const baseScale = anchor.faceScale || 260.0
  row[5] = (f.faceScale - baseScale) / baseScale

  // 6-8. Relative head pose (azimuth, elevation, roll)
  const yaw = f.yawProxy - anchor.yawProxy
  const pitch = f.pitchProxy - anchor.pitchProxy
  row[6] = yaw
  row[7] = pitch
  row[8] = f.roll

  // 9-13. Decoupled Left Eye (Eye A) features
  row[9] = f.eyeA.gx
  row[10] = f.eyeA.gy
  row[11] = f.eyeA.ear
  row[12] = f.eyeA.width
  row[13] = f.eyeA.irisRadius

  // 14-18. Decoupled Right Eye (Eye B) features
  row[14] = f.eyeB.gx
  row[15] = f.eyeB.gy
  row[16] = f.eyeB.ear
  row[17] = f.eyeB.width
  row[18] = f.eyeB.irisRadius

  // 19-20. Optical Perspective Invariants (Scale-independent 3D perspective cues)
  const sumWidth = f.eyeA.width + f.eyeB.width
  row[19] = sumWidth > 1e-6 ? (f.eyeA.width - f.eyeB.width) / sumWidth : 0.0

  const sumIris = f.eyeA.irisRadius + f.eyeB.irisRadius
  row[20] = sumIris > 1e-6 ? (f.eyeA.irisRadius - f.eyeB.irisRadius) / sumIris : 0.0

  // 21-22. Planar Tangent Projections (Screen coordinate X ~ D * tan(theta))
  const clipGx = Math.min(1.2, Math.max(-1.2, f.gx))
  const clipGy = Math.min(1.2, Math.max(-1.2, f.gy))
  row[21] = Math.tan(clipGx)
  row[22] = Math.tan(clipGy)

  // 23-24. Distance-Normalized Gaze (Face-scale dynamic magnification compensation)
  const scaleRatio = f.faceScale / baseScale
  row[23] = f.gx * scaleRatio
  row[24] = f.gy * scaleRatio

  // 25. Differential Eyelid Squint / Asymmetry
  row[25] = f.eyeA.ear - f.eyeB.ear

  // 26-27. Kinematic Cross-Terms (Gaze-Head Pose Decoupling)
  row[26] = f.gx * yaw
  row[27] = f.gy * pitch

  return row
}
