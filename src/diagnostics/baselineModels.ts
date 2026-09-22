/**
 * baselineModels.ts — Comparative Baselines & Numerical Condition Number Analysis.
 */

import type { RawCalibrationSample } from '../core/modelSelection'
import {
  buildEnhancedFeatureRow,
  computeCalibrationAnchor,
  ENHANCED_FEATURE_COUNT,
} from '../core/features28'
import {
  filterSaccadicTransit,
  aggregateTargetCentroids,
} from '../core/modelSelection'
import {
  solveElasticNet,
  solveRidge,
  standardizeFeatures,
} from '../core/solvers'

export interface ModelBenchmarkRow {
  modelId: string
  modelName: string
  cvRmsPx: number
  p95ErrorPx: number
  cvErrorXPx: number
  cvErrorYPx: number
  conditionNumber: number
  latencyMs: number
}

export function estimateConditionNumber(A: Float64Array[], p: number): number {
  const n = A.length
  if (n === 0 || p === 0) return 1.0

  const AtA: number[][] = Array.from({ length: p }, () => new Array(p).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      for (let k = j; k < p; k++) {
        AtA[j][k] += A[i][j] * A[i][k]
      }
    }
  }
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < j; k++) {
      AtA[j][k] = AtA[k][j]
    }
  }

  let v = new Float64Array(p).fill(1.0 / Math.sqrt(p))
  let lambdaMax = 1.0
  for (let iter = 0; iter < 50; iter++) {
    const w = new Float64Array(p)
    for (let j = 0; j < p; j++) {
      let sum = 0
      for (let k = 0; k < p; k++) sum += AtA[j][k] * v[k]
      w[j] = sum
    }
    const norm = Math.hypot(...w)
    if (norm < 1e-12) break
    lambdaMax = norm
    for (let j = 0; j < p; j++) v[j] = w[j] / norm
  }

  let minDiag = Infinity
  for (let j = 0; j < p; j++) {
    if (AtA[j][j] < minDiag) minDiag = Math.max(1e-12, AtA[j][j])
  }

  return lambdaMax / minDiag
}

/**
 * 1. Classical Polynomial OLS (WebGazer-style) on raw unaggregated frames
 */
export function evaluateClassicalPolyOLS(samples: RawCalibrationSample[]): ModelBenchmarkRow {
  const p = 6
  const groups = samples.map((s) => s.group)
  const uniqueGroups = Array.from(new Set(groups))
  const errorsX: number[] = [], errorsY: number[] = [], errorsEuc: number[] = []

  const t0 = performance.now()
  const fullX: Float64Array[] = samples.map((s) => {
    const ix = s.feature.gx + 0.5, iy = s.feature.gy + 0.5
    const r = new Float64Array(6)
    r[0] = 1.0; r[1] = ix; r[2] = iy; r[3] = ix * iy; r[4] = ix * ix; r[5] = iy * iy
    return r
  })
  const condNum = estimateConditionNumber(fullX, 6)

  for (const g of uniqueGroups) {
    const trIndices: number[] = []
    const teIndices: number[] = []
    for (let i = 0; i < samples.length; i++) {
      if (groups[i] === g) teIndices.push(i)
      else trIndices.push(i)
    }

    const XTr = trIndices.map((i) => fullX[i])
    const YxTr = trIndices.map((i) => samples[i].target.x)
    const YyTr = trIndices.map((i) => samples[i].target.y)

    // Unregularized OLS (lambda 1e-9)
    const wX = solveRidge(XTr, YxTr, 1e-9)
    const wY = solveRidge(XTr, YyTr, 1e-9)

    for (const teIdx of teIndices) {
      const row = fullX[teIdx]
      let px = 0, py = 0
      for (let j = 0; j < 6; j++) {
        px += row[j] * wX[j]
        py += row[j] * wY[j]
      }
      const ex = Math.abs(px - samples[teIdx].target.x)
      const ey = Math.abs(py - samples[teIdx].target.y)
      errorsX.push(ex); errorsY.push(ey); errorsEuc.push(Math.hypot(ex, ey))
    }
  }

  const latency = (performance.now() - t0) / samples.length
  const rms = (a: number[]) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / Math.max(1, a.length))
  const sorted = [...errorsEuc].sort((a, b) => a - b)

  return {
    modelId: 'B1-WebGazer-Poly',
    modelName: 'Classical Polynomial (WebGazer OLS)',
    cvRmsPx: rms(errorsEuc),
    p95ErrorPx: sorted[Math.floor(sorted.length * 0.95)] || 0,
    cvErrorXPx: rms(errorsX),
    cvErrorYPx: rms(errorsY),
    conditionNumber: condNum,
    latencyMs: latency,
  }
}

/**
 * 2. Linear Ridge: [1, gx, gy]
 */
export function evaluateLinearRidge(samples: RawCalibrationSample[]): ModelBenchmarkRow {
  const p = 3
  const groups = samples.map((s) => s.group)
  const uniqueGroups = Array.from(new Set(groups))
  const errorsX: number[] = [], errorsY: number[] = [], errorsEuc: number[] = []

  const t0 = performance.now()
  const fullX: Float64Array[] = samples.map((s) => {
    const r = new Float64Array(3)
    r[0] = 1.0; r[1] = s.feature.gx; r[2] = s.feature.gy
    return r
  })
  const condNum = estimateConditionNumber(fullX, 3)

  for (const g of uniqueGroups) {
    const trIndices: number[] = []
    const teIndices: number[] = []
    for (let i = 0; i < samples.length; i++) {
      if (groups[i] === g) teIndices.push(i)
      else trIndices.push(i)
    }

    const XTr = trIndices.map((i) => fullX[i])
    const YxTr = trIndices.map((i) => samples[i].target.x)
    const YyTr = trIndices.map((i) => samples[i].target.y)

    const { XStd, mu, sd } = standardizeFeatures(XTr)
    const wX = solveRidge(XStd, YxTr, 1e-3)
    const wY = solveRidge(XStd, YyTr, 1e-3)

    for (const teIdx of teIndices) {
      const row = fullX[teIdx]
      let px = wX[0], py = wY[0]
      for (let j = 1; j < 3; j++) {
        const z = (row[j] - mu[j - 1]) / sd[j - 1]
        px += z * wX[j]
        py += z * wY[j]
      }
      const ex = Math.abs(px - samples[teIdx].target.x)
      const ey = Math.abs(py - samples[teIdx].target.y)
      errorsX.push(ex); errorsY.push(ey); errorsEuc.push(Math.hypot(ex, ey))
    }
  }

  const latency = (performance.now() - t0) / samples.length
  const rms = (a: number[]) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / Math.max(1, a.length))
  const sorted = [...errorsEuc].sort((a, b) => a - b)

  return {
    modelId: 'B2-Ridge-Linear',
    modelName: 'Standard Linear Ridge',
    cvRmsPx: rms(errorsEuc),
    p95ErrorPx: sorted[Math.floor(sorted.length * 0.95)] || 0,
    cvErrorXPx: rms(errorsX),
    cvErrorYPx: rms(errorsY),
    conditionNumber: condNum,
    latencyMs: latency,
  }
}

/**
 * 3. Standard Polynomial Ridge: [1, gx, gy, gx*gy, gx^2, gy^2]
 */
export function evaluatePolynomialRidge(samples: RawCalibrationSample[]): ModelBenchmarkRow {
  const p = 6
  const groups = samples.map((s) => s.group)
  const uniqueGroups = Array.from(new Set(groups))
  const errorsX: number[] = [], errorsY: number[] = [], errorsEuc: number[] = []

  const t0 = performance.now()
  const fullX: Float64Array[] = samples.map((s) => {
    const gx = s.feature.gx, gy = s.feature.gy
    const r = new Float64Array(6)
    r[0] = 1.0; r[1] = gx; r[2] = gy; r[3] = gx * gy; r[4] = gx * gx; r[5] = gy * gy
    return r
  })
  const condNum = estimateConditionNumber(fullX, 6)

  for (const g of uniqueGroups) {
    const trIndices: number[] = []
    const teIndices: number[] = []
    for (let i = 0; i < samples.length; i++) {
      if (groups[i] === g) teIndices.push(i)
      else trIndices.push(i)
    }

    const XTr = trIndices.map((i) => fullX[i])
    const YxTr = trIndices.map((i) => samples[i].target.x)
    const YyTr = trIndices.map((i) => samples[i].target.y)

    const { XStd, mu, sd } = standardizeFeatures(XTr)
    const wX = solveRidge(XStd, YxTr, 1e-3)
    const wY = solveRidge(XStd, YyTr, 1e-3)

    for (const teIdx of teIndices) {
      const row = fullX[teIdx]
      let px = wX[0], py = wY[0]
      for (let j = 1; j < 6; j++) {
        const z = (row[j] - mu[j - 1]) / sd[j - 1]
        px += z * wX[j]
        py += z * wY[j]
      }
      const ex = Math.abs(px - samples[teIdx].target.x)
      const ey = Math.abs(py - samples[teIdx].target.y)
      errorsX.push(ex); errorsY.push(ey); errorsEuc.push(Math.hypot(ex, ey))
    }
  }

  const latency = (performance.now() - t0) / samples.length
  const rms = (a: number[]) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / Math.max(1, a.length))
  const sorted = [...errorsEuc].sort((a, b) => a - b)

  return {
    modelId: 'B3-Ridge-Poly2',
    modelName: 'Standard Polynomial Ridge',
    cvRmsPx: rms(errorsEuc),
    p95ErrorPx: sorted[Math.floor(sorted.length * 0.95)] || 0,
    cvErrorXPx: rms(errorsX),
    cvErrorYPx: rms(errorsY),
    conditionNumber: condNum,
    latencyMs: latency,
  }
}

/**
 * 4. Production Per-Eye Ridge Baseline
 */
export function evaluatePerEyeRidge(samples: RawCalibrationSample[]): ModelBenchmarkRow {
  const p = 8
  const groups = samples.map((s) => s.group)
  const uniqueGroups = Array.from(new Set(groups))
  const errorsX: number[] = [], errorsY: number[] = [], errorsEuc: number[] = []

  const t0 = performance.now()
  const fullX: Float64Array[] = samples.map((s) => {
    const r = new Float64Array(8)
    r[0] = 1.0; r[1] = s.feature.gx; r[2] = s.feature.gy
    r[3] = s.feature.eyeA.gx; r[4] = s.feature.eyeA.gy
    r[5] = s.feature.eyeB.gx; r[6] = s.feature.eyeB.gy
    r[7] = s.feature.disparity
    return r
  })
  const condNum = estimateConditionNumber(fullX, 8)

  for (const g of uniqueGroups) {
    const trIndices: number[] = []
    const teIndices: number[] = []
    for (let i = 0; i < samples.length; i++) {
      if (groups[i] === g) teIndices.push(i)
      else trIndices.push(i)
    }

    const XTr = trIndices.map((i) => fullX[i])
    const YxTr = trIndices.map((i) => samples[i].target.x)
    const YyTr = trIndices.map((i) => samples[i].target.y)

    const { XStd, mu, sd } = standardizeFeatures(XTr)
    const wX = solveRidge(XStd, YxTr, 1e-3)
    const wY = solveRidge(XStd, YyTr, 1e-3)

    for (const teIdx of teIndices) {
      const row = fullX[teIdx]
      let px = wX[0], py = wY[0]
      for (let j = 1; j < 8; j++) {
        const z = (row[j] - mu[j - 1]) / sd[j - 1]
        px += z * wX[j]
        py += z * wY[j]
      }
      const ex = Math.abs(px - samples[teIdx].target.x)
      const ey = Math.abs(py - samples[teIdx].target.y)
      errorsX.push(ex); errorsY.push(ey); errorsEuc.push(Math.hypot(ex, ey))
    }
  }

  const latency = (performance.now() - t0) / samples.length
  const rms = (a: number[]) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / Math.max(1, a.length))
  const sorted = [...errorsEuc].sort((a, b) => a - b)

  return {
    modelId: 'B4-Ridge-PerEye',
    modelName: 'Production Per-Eye Ridge',
    cvRmsPx: rms(errorsEuc),
    p95ErrorPx: sorted[Math.floor(sorted.length * 0.95)] || 0,
    cvErrorXPx: rms(errorsX),
    cvErrorYPx: rms(errorsY),
    conditionNumber: condNum,
    latencyMs: latency,
  }
}

/**
 * 5. Proposed OpenGaze 28-Term ElasticNet with Saccadic Transit Discard + Centroids
 */
export function evaluateProposedOpenGaze(rawSamples: RawCalibrationSample[]): ModelBenchmarkRow {
  const t0 = performance.now()
  const cleaned = filterSaccadicTransit(rawSamples, 3)
  const centroids = aggregateTargetCentroids(cleaned, 0.2)
  const anchor = computeCalibrationAnchor(centroids)

  const p = ENHANCED_FEATURE_COUNT
  const fullX: Float64Array[] = centroids.map((s) => buildEnhancedFeatureRow(s.feature, anchor))
  const condNum = estimateConditionNumber(fullX, p)

  const groups = centroids.map((s) => s.group)
  const uniqueGroups = Array.from(new Set(groups))
  const errorsX: number[] = [], errorsY: number[] = [], errorsEuc: number[] = []

  for (const g of uniqueGroups) {
    const trIndices: number[] = []
    const teIndices: number[] = []
    for (let i = 0; i < centroids.length; i++) {
      if (groups[i] === g) teIndices.push(i)
      else trIndices.push(i)
    }

    const XTr = trIndices.map((i) => fullX[i])
    const YxTr = trIndices.map((i) => centroids[i].target.x)
    const YyTr = trIndices.map((i) => centroids[i].target.y)

    const { XStd, mu, sd } = standardizeFeatures(XTr)
    const wX = solveElasticNet(XStd, YxTr, 0.3, 1.5e-3)
    const wY = solveElasticNet(XStd, YyTr, 0.3, 1.2e-3)

    for (const teIdx of teIndices) {
      const row = fullX[teIdx]
      let px = wX[0], py = wY[0]
      for (let j = 1; j < p; j++) {
        const z = (row[j] - mu[j - 1]) / sd[j - 1]
        px += z * wX[j]
        py += z * wY[j]
      }
      const ex = Math.abs(px - centroids[teIdx].target.x)
      const ey = Math.abs(py - centroids[teIdx].target.y)
      errorsX.push(ex); errorsY.push(ey); errorsEuc.push(Math.hypot(ex, ey))
    }
  }

  const latency = (performance.now() - t0) / centroids.length
  const rms = (a: number[]) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / Math.max(1, a.length))
  const sorted = [...errorsEuc].sort((a, b) => a - b)

  return {
    modelId: 'Proposed-OpenGaze-28Term',
    modelName: 'OpenGaze (Proposed 28-Term ElasticNet)',
    cvRmsPx: rms(errorsEuc),
    p95ErrorPx: sorted[Math.floor(sorted.length * 0.95)] || 0,
    cvErrorXPx: rms(errorsX),
    cvErrorYPx: rms(errorsY),
    conditionNumber: condNum,
    latencyMs: latency,
  }
}
