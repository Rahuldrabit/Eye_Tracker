/**
 * modelSelection.ts — Grouped Leave-One-Target-Out Cross-Validation & Model Selection.
 */

import type { EyeFeature, CalibrationAnchor } from './features28'
import { buildEnhancedFeatureRow, computeCalibrationAnchor, ENHANCED_FEATURE_COUNT } from './features28'
import { solveElasticNet, standardizeFeatures } from './solvers'

export interface RawCalibrationSample {
  group: number
  target: { x: number; y: number }
  feature: EyeFeature
  timestamp: number
}

export interface CVEvaluationResult {
  cvErrorPx: number
  cvP95Px: number
  cvErrorXPx: number
  cvErrorYPx: number
  targetCount: number
  sampleCount: number
}

export function filterSaccadicTransit(
  samples: RawCalibrationSample[],
  discardFrames = 3
): RawCalibrationSample[] {
  const groups = new Map<number, RawCalibrationSample[]>()
  for (const s of samples) {
    if (!groups.has(s.group)) groups.set(s.group, [])
    groups.get(s.group)!.push(s)
  }

  const cleaned: RawCalibrationSample[] = []
  for (const group of groups.values()) {
    if (group.length > discardFrames) {
      cleaned.push(...group.slice(discardFrames))
    } else {
      cleaned.push(...group)
    }
  }
  return cleaned
}

export function aggregateTargetCentroids(
  samples: RawCalibrationSample[],
  trimRatio = 0.2
): RawCalibrationSample[] {
  const groups = new Map<number, RawCalibrationSample[]>()
  for (const s of samples) {
    if (!groups.has(s.group)) groups.set(s.group, [])
    groups.get(s.group)!.push(s)
  }

  const centroids: RawCalibrationSample[] = []
  for (const [group, grp] of groups.entries()) {
    if (grp.length === 0) continue

    const trim = (vals: number[]): number => {
      vals.sort((a, b) => a - b)
      const k = Math.floor(vals.length * trimRatio)
      const sub = vals.slice(k, vals.length - k)
      return sub.reduce((a, b) => a + b, 0) / Math.max(1, sub.length)
    }

    const base = grp[0]
    const avgFeature: EyeFeature = {
      gx: trim(grp.map((s) => s.feature.gx)),
      gy: trim(grp.map((s) => s.feature.gy)),
      disparity: trim(grp.map((s) => s.feature.disparity)),
      ear: trim(grp.map((s) => s.feature.ear)),
      faceScale: trim(grp.map((s) => s.feature.faceScale)),
      yawProxy: trim(grp.map((s) => s.feature.yawProxy)),
      pitchProxy: trim(grp.map((s) => s.feature.pitchProxy)),
      roll: trim(grp.map((s) => s.feature.roll)),
      noseX: trim(grp.map((s) => s.feature.noseX)),
      noseY: trim(grp.map((s) => s.feature.noseY)),
      eyeA: {
        gx: trim(grp.map((s) => s.feature.eyeA.gx)),
        gy: trim(grp.map((s) => s.feature.eyeA.gy)),
        ear: trim(grp.map((s) => s.feature.eyeA.ear)),
        width: trim(grp.map((s) => s.feature.eyeA.width)),
        irisRadius: trim(grp.map((s) => s.feature.eyeA.irisRadius)),
      },
      eyeB: {
        gx: trim(grp.map((s) => s.feature.eyeB.gx)),
        gy: trim(grp.map((s) => s.feature.eyeB.gy)),
        ear: trim(grp.map((s) => s.feature.eyeB.ear)),
        width: trim(grp.map((s) => s.feature.eyeB.width)),
        irisRadius: trim(grp.map((s) => s.feature.eyeB.irisRadius)),
      },
    }

    centroids.push({
      group,
      target: base.target,
      feature: avgFeature,
      timestamp: base.timestamp,
    })
  }

  return centroids
}
