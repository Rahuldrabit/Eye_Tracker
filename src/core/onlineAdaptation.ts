/**
 * onlineAdaptation.ts — Continuous In-Situ Fine-Tuning & Drift Adaptation Engine.
 *
 * THE PROBLEM:
 * Single-webcam eye tracking accuracy steadily decays within 3-5 minutes as the
 * user naturally slouches, changes posture, shifts chair height, or experiences
 * pupil dilation changes.
 *
 * THE SOLUTION:
 * 1. Interaction-Anchored Supervision:
 *    When a user clicks or taps a UI target, human eye-hand coordination mandates
 *    that the eyes foveate the target ~80-250ms prior to the physical event.
 *    This provides free supervised pairs (feature_vector, target_coord) passively.
 *
 * 2. Bayesian Prior Anchor (Prevents Catastrophic Divergence):
 *    Naive online least-squares quickly diverges if interactions cluster in one
 *    screen quadrant (e.g. reading a single column). We formulate continuous
 *    fine-tuning with an elastic penalty anchored to the base calibration W_0:
 *
 *    min_W  ||Y - X W||_2^2 + lambda * ||W||_2^2 + gamma * ||W - W_0||_2^2
 *
 *    The anchor parameter gamma > 0 guarantees that weights adapt locally while
 *    maintaining global stability across the entire display.
 *
 * 3. Robust Residual Gating:
 *    Prunes false clicks (muscle memory clicks where the eye moved early) using
 *    Median Absolute Deviation (MAD) thresholding.
 */

import type { EyeFeature, CalibrationAnchor } from './features28'
import { buildEnhancedFeatureRow, ENHANCED_FEATURE_COUNT } from './features28'
import { solveRidge, standardizeFeatures } from './solvers'

export interface InteractionSample {
  feature: EyeFeature
  target: { x: number; y: number }
  timestamp: number
}

export interface AdaptationConfig {
  maxWindowSize: number      // Max samples retained in sliding window (e.g., 50)
  minSamplesToAdapt: number  // Minimum samples before triggering fine-tune (e.g., 8)
  featureLeadWindowMs: [number, number] // [oldest, newest] ms before click (e.g. [220, 60])
  residualMadLimit: number   // Outlier threshold in MAD units (default: 2.5)
  adaptEveryNSamples: number // Refit frequency (default: 3)
  anchorWeightGamma: number  // Elastic strength tying model to W_0 (default: 5.0)
  ridgeLambda: number        // Standard L2 regularizer (default: 1e-3)
}

export const DEFAULT_ADAPTATION_CONFIG: AdaptationConfig = {
  maxWindowSize: 60,
  minSamplesToAdapt: 10,
  featureLeadWindowMs: [240, 60],
  residualMadLimit: 2.5,
  adaptEveryNSamples: 3,
  anchorWeightGamma: 5.0,
  ridgeLambda: 1e-3,
}

export class ContinuousAdaptationEngine {
  private config: AdaptationConfig
  private anchor: CalibrationAnchor
  private w0X: Float64Array
  private w0Y: Float64Array
  private currentWX: Float64Array
  private currentWY: Float64Array
  private rollingFeatures: { feature: EyeFeature; timestamp: number }[] = []
  private samples: InteractionSample[] = []
  private samplesSinceLastRefit = 0

  constructor(
    anchor: CalibrationAnchor,
    initialWeightsX: Float64Array,
    initialWeightsY: Float64Array,
    config: Partial<AdaptationConfig> = {}
  ) {
    this.config = { ...DEFAULT_ADAPTATION_CONFIG, ...config }
    this.anchor = anchor
    this.w0X = new Float64Array(initialWeightsX)
    this.w0Y = new Float64Array(initialWeightsY)
    this.currentWX = new Float64Array(initialWeightsX)
    this.currentWY = new Float64Array(initialWeightsY)
  }

  /**
   * Pushes incoming high-frequency eye features into the rolling pre-interaction buffer.
   */
  pushFrame(feature: EyeFeature, timestamp: number = Date.now()): void {
    this.rollingFeatures.push({ feature, timestamp })
    // Retain past 1 second of frames
    const cutoff = timestamp - 1000
    while (this.rollingFeatures.length > 0 && this.rollingFeatures[0].timestamp < cutoff) {
      this.rollingFeatures.shift()
    }
  }

  /**
   * Registers a user interaction (mouse click, touch tap) at known screen coordinates.
   * Extracts the foveated feature immediately preceding the physical event.
   */
  registerInteraction(targetX: number, targetY: number, clickTime: number = Date.now()): boolean {
    const [oldestLead, newestLead] = this.config.featureLeadWindowMs
    const tStart = clickTime - oldestLead
    const tEnd = clickTime - newestLead

    const candidates = this.rollingFeatures.filter(
      (f) => f.timestamp >= tStart && f.timestamp <= tEnd
    )

    if (candidates.length === 0) return false

    // Average the features in the pre-click foveation window
    const avgFeature = this.averageFeatures(candidates.map((c) => c.feature))

    this.samples.push({
      feature: avgFeature,
      target: { x: targetX, y: targetY },
      timestamp: clickTime,
    })

    // Evict oldest if exceeding sliding window size
    if (this.samples.length > this.config.maxWindowSize) {
      this.samples.shift()
    }

    this.samplesSinceLastRefit++

    if (
      this.samples.length >= this.config.minSamplesToAdapt &&
      this.samplesSinceLastRefit >= this.config.adaptEveryNSamples
    ) {
      this.performAdaptiveFineTune()
      this.samplesSinceLastRefit = 0
      return true
    }

    return false
  }

  /**
   * Solves the Bayesian-anchored fine-tuning objective across interaction samples.
   */
  private performAdaptiveFineTune(): void {
    const p = ENHANCED_FEATURE_COUNT
    const n = this.samples.length

    // Build design matrix from current samples
    const X: Float64Array[] = this.samples.map((s) =>
      buildEnhancedFeatureRow(s.feature, this.anchor)
    )
    const yX = new Float64Array(n)
    const yY = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      yX[i] = this.samples[i].target.x
      yY[i] = this.samples[i].target.y
    }

    // Standardize
    const { XStd: Z } = standardizeFeatures(X)

    // Two-pass robust filtering: drop samples whose residual exceeds MAD limit
    const predX = this.predictBatch(Z, this.currentWX)
    const predY = this.predictBatch(Z, this.currentWY)
    const residuals = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      residuals[i] = Math.hypot(yX[i] - predX[i], yY[i] - predY[i])
    }

    const medRes = this.median(residuals)
    const mad = 1.4826 * this.median(residuals.map((r) => Math.abs(r - medRes))) || 1.0
    const cutoff = medRes + this.config.residualMadLimit * mad

    const validIndices: number[] = []
    for (let i = 0; i < n; i++) {
      if (residuals[i] <= cutoff) validIndices.push(i)
    }

    if (validIndices.length < Math.max(5, p / 3)) return // Not enough valid points

    const cleanZ = validIndices.map((i) => Z[i])
    const cleanYX = new Float64Array(validIndices.map((i) => yX[i]))
    const cleanYY = new Float64Array(validIndices.map((i) => yY[i]))

    // Solve with Bayesian Prior Anchor W_0:
    // (Z^T Z + (lambda + gamma) S) W = Z^T Y + gamma * S * W_0
    const gamma = this.config.anchorWeightGamma
    const lambda = this.config.ridgeLambda

    this.currentWX = this.solveAnchored(cleanZ, cleanYX, this.w0X, p, lambda, gamma)
    this.currentWY = this.solveAnchored(cleanZ, cleanYY, this.w0Y, p, lambda, gamma)
  }

  private solveAnchored(
    Z: Float64Array[],
    y: Float64Array,
    w0: Float64Array,
    p: number,
    lambda: number,
    gamma: number
  ): Float64Array {
    const n = Z.length
    const AtA: number[][] = Array.from({ length: p }, () => new Array(p).fill(0))
    const Aty = new Float64Array(p)

    for (let i = 0; i < n; i++) {
      const row = Z[i]
      const yi = y[i]
      for (let j = 0; j < p; j++) {
        Aty[j] += row[j] * yi
        for (let k = j; k < p; k++) {
          AtA[j][k] += row[j] * row[k]
        }
      }
    }

    for (let j = 0; j < p; j++) {
      for (let k = 0; k < j; k++) {
        AtA[j][k] = AtA[k][j]
      }
    }

    // Add penalties and anchor to RHS
    for (let j = 1; j < p; j++) {
      AtA[j][j] += lambda + gamma
      Aty[j] += gamma * w0[j]
    }

    return solveRidge(Z, y, p, lambda + gamma)
  }

  getWeights(): { weightsX: Float64Array; weightsY: Float64Array } {
    return {
      weightsX: new Float64Array(this.currentWX),
      weightsY: new Float64Array(this.currentWY),
    }
  }

  private predictBatch(Z: Float64Array[], w: Float64Array): Float64Array {
    const n = Z.length
    const p = w.length
    const out = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let sum = 0
      for (let j = 0; j < p; j++) {
        sum += Z[i][j] * w[j]
      }
      out[i] = sum
    }
    return out
  }

  private median(arr: Float64Array | number[]): number {
    const s = Array.from(arr).sort((a, b) => a - b)
    if (s.length === 0) return 0
    const m = s.length >> 1
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2.0
  }

  private averageFeatures(features: EyeFeature[]): EyeFeature {
    const n = features.length
    const base = features[0]
    let sumGx = 0, sumGy = 0, sumDisp = 0, sumEar = 0, sumScale = 0
    let sumYaw = 0, sumPitch = 0, sumRoll = 0, sumNoseX = 0, sumNoseY = 0

    for (const f of features) {
      sumGx += f.gx
      sumGy += f.gy
      sumDisp += f.disparity
      sumEar += f.ear
      sumScale += f.faceScale
      sumYaw += f.yawProxy
      sumPitch += f.pitchProxy
      sumRoll += f.roll
      sumNoseX += f.noseX
      sumNoseY += f.noseY
    }

    return {
      gx: sumGx / n,
      gy: sumGy / n,
      disparity: sumDisp / n,
      ear: sumEar / n,
      faceScale: sumScale / n,
      yawProxy: sumYaw / n,
      pitchProxy: sumPitch / n,
      roll: sumRoll / n,
      noseX: sumNoseX / n,
      noseY: sumNoseY / n,
      eyeA: { ...base.eyeA },
      eyeB: { ...base.eyeB },
    }
  }
}
