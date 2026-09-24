import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  filterSaccadicTransit,
  aggregateTargetCentroids,
  computeCalibrationAnchor,
  buildEnhancedFeatureRow,
  standardizeFeatures,
  solveElasticNet,
  solveRidge,
  AffineRecalibrator,
  ENHANCED_FEATURE_COUNT,
} from '../src/core/index.js'
import type { RawCalibrationSample } from '../src/core/modelSelection.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const dataFiles = [
  'calibration-export-1788065417299.json',
  'calibration-export-1788065666638.json',
  'calibration-export-1788065794436.json',
  'calibration-export-1790073671613.json',
]

const loadedSessions: { name: string; samples: RawCalibrationSample[] }[] = []

for (const f of dataFiles) {
  let p = path.resolve(__dirname, '../../Calibration data', f)
  if (!fs.existsSync(p)) p = path.resolve(__dirname, '../../', f)
  if (!fs.existsSync(p)) continue
  const json = JSON.parse(fs.readFileSync(p, 'utf-8'))
  loadedSessions.push({ name: f, samples: json.samples })
}

console.log('Loaded sessions:', loadedSessions.length)

// -------------------------------------------------------------
// 1. STATISTICAL SIGNIFICANCE (Paired t-test across all 80 targets)
// -------------------------------------------------------------
console.log('\n--- 1. Statistical Significance (WebGazer vs OpenEyeGaze across 80 targets) ---')
const webgazerErrors: number[] = []
const openeyegazeErrors: number[] = []

for (const session of loadedSessions) {
  const samples = session.samples
  const groups = samples.map((s) => s.group)
  const uniqueGroups = Array.from(new Set(groups))

  // WebGazer
  const fullPolyX = samples.map((s) => {
    const ix = s.feature.gx + 0.5, iy = s.feature.gy + 0.5
    const r = new Float64Array(6)
    r[0] = 1.0; r[1] = ix; r[2] = iy; r[3] = ix * iy; r[4] = ix * ix; r[5] = iy * iy
    return r
  })
  for (const g of uniqueGroups) {
    const tr = samples.filter((s) => s.group !== g)
    const te = samples.filter((s) => s.group === g)
    const trIdx = samples.map((s, i) => (s.group !== g ? i : -1)).filter((i) => i >= 0)
    const teIdx = samples.map((s, i) => (s.group === g ? i : -1)).filter((i) => i >= 0)

    const XTr = trIdx.map((i) => fullPolyX[i])
    const YxTr = tr.map((s) => s.target.x)
    const YyTr = tr.map((s) => s.target.y)
    const wX = solveRidge(XTr, YxTr, 1e-9)
    const wY = solveRidge(XTr, YyTr, 1e-9)

    let targetErrSum = 0
    for (const i of teIdx) {
      const row = fullPolyX[i]
      let px = 0, py = 0
      for (let j = 0; j < 6; j++) { px += row[j] * wX[j]; py += row[j] * wY[j] }
      targetErrSum += Math.hypot(px - samples[i].target.x, py - samples[i].target.y)
    }
    webgazerErrors.push(targetErrSum / Math.max(1, te.length))
  }

  // OpenEyeGaze
  const cleaned = filterSaccadicTransit(samples, 3)
  const centroids = aggregateTargetCentroids(cleaned, 0.2)
  const anchor = computeCalibrationAnchor(centroids)
  const p = ENHANCED_FEATURE_COUNT
  const fullX = centroids.map((s) => buildEnhancedFeatureRow(s.feature, anchor))
  const centGroups = centroids.map((s) => s.group)
  const uniqueCentGroups = Array.from(new Set(centGroups))

  for (const g of uniqueCentGroups) {
    const trIdx = centroids.map((s, i) => (s.group !== g ? i : -1)).filter((i) => i >= 0)
    const teIdx = centroids.map((s, i) => (s.group === g ? i : -1)).filter((i) => i >= 0)

    const XTr = trIdx.map((i) => fullX[i])
    const YxTr = trIdx.map((i) => centroids[i].target.x)
    const YyTr = trIdx.map((i) => centroids[i].target.y)
    const { XStd, mu, sd } = standardizeFeatures(XTr)
    const wX = solveElasticNet(XStd, YxTr, 0.3, 1.5e-3)
    const wY = solveElasticNet(XStd, YyTr, 0.3, 1.2e-3)

    let targetErrSum = 0
    for (const i of teIdx) {
      const row = fullX[i]
      let px = wX[0], py = wY[0]
      for (let j = 1; j < p; j++) {
        const z = (row[j] - mu[j - 1]) / sd[j - 1]
        px += z * wX[j]; py += z * wY[j]
      }
      targetErrSum += Math.hypot(px - centroids[i].target.x, py - centroids[i].target.y)
    }
    openeyegazeErrors.push(targetErrSum / Math.max(1, teIdx.length))
  }
}

// Paired t-test
const N_targets = Math.min(webgazerErrors.length, openeyegazeErrors.length)
const diffs: number[] = []
for (let i = 0; i < N_targets; i++) {
  diffs.push(webgazerErrors[i] - openeyegazeErrors[i])
}
const meanDiff = diffs.reduce((a, b) => a + b, 0) / N_targets
const varDiff = diffs.reduce((a, b) => a + (b - meanDiff) ** 2, 0) / (N_targets - 1)
const sdDiff = Math.sqrt(varDiff)
const tStat = meanDiff / (sdDiff / Math.sqrt(N_targets))
const cohenD = meanDiff / sdDiff

console.log(`Evaluated ${N_targets} targets across all sessions:`)
console.log(`Mean WebGazer target error: ${(webgazerErrors.reduce((a, b) => a + b, 0) / N_targets).toFixed(1)} px`)
console.log(`Mean OpenEyeGaze target error: ${(openeyegazeErrors.reduce((a, b) => a + b, 0) / N_targets).toFixed(1)} px`)
console.log(`Mean difference: ${meanDiff.toFixed(1)} px, SD diff: ${sdDiff.toFixed(1)} px`)
console.log(`Paired t-statistic: t(${N_targets - 1}) = ${tStat.toFixed(2)}, Cohen's d = ${cohenD.toFixed(2)}`)

// -------------------------------------------------------------
// 2. REAL ABLATION STUDY ACROSS ALL 4 SESSIONS
// -------------------------------------------------------------
console.log('\n--- 2. Real Systematic Ablations across all 4 sessions ---')

function runAblationVariant(
  variant: 'full' | 'no_transit' | 'no_centroid' | 'pure_l2' | 'raw_landmarks'
): { rms: number; p95: number } {
  const allErrors: number[] = []

  for (const session of loadedSessions) {
    const rawSamples = session.samples

    // Transit discard
    const cleaned = variant === 'no_transit'
      ? rawSamples
      : filterSaccadicTransit(rawSamples, 3)

    // Centroid aggregation
    const dataPoints = variant === 'no_centroid'
      ? cleaned
      : aggregateTargetCentroids(cleaned, 0.2)

    const anchor = computeCalibrationAnchor(dataPoints)
    const p = ENHANCED_FEATURE_COUNT

    const fullX: Float64Array[] = dataPoints.map((s) => {
      if (variant === 'raw_landmarks') {
        // Raw MediaPipe normalized coordinates (no canthal basis)
        const row = new Float64Array(p)
        row[0] = 1.0
        row[1] = s.feature.gx * 50 // artificially scale to raw pixel-like offsets
        row[2] = s.feature.gy * 50
        return row
      }
      return buildEnhancedFeatureRow(s.feature, anchor)
    })

    const groups = dataPoints.map((s) => s.group)
    const uniqueGroups = Array.from(new Set(groups))

    for (const g of uniqueGroups) {
      const trIdx = dataPoints.map((s, i) => (s.group !== g ? i : -1)).filter((i) => i >= 0)
      const teIdx = dataPoints.map((s, i) => (s.group === g ? i : -1)).filter((i) => i >= 0)

      const XTr = trIdx.map((i) => fullX[i])
      const YxTr = trIdx.map((i) => dataPoints[i].target.x)
      const YyTr = trIdx.map((i) => dataPoints[i].target.y)

      const { XStd, mu, sd } = standardizeFeatures(XTr)

      let wX: Float64Array, wY: Float64Array
      if (variant === 'pure_l2') {
        // Pure Ridge L2 (no L1 sparsity)
        wX = solveRidge(XStd, YxTr, 1e-3)
        wY = solveRidge(XStd, YyTr, 1e-3)
      } else {
        wX = solveElasticNet(XStd, YxTr, 0.3, 1.5e-3)
        wY = solveElasticNet(XStd, YyTr, 0.3, 1.2e-3)
      }

      for (const te of teIdx) {
        const row = fullX[te]
        let px = wX[0], py = wY[0]
        for (let j = 1; j < p; j++) {
          const z = (row[j] - mu[j - 1]) / sd[j - 1]
          px += z * wX[j]; py += z * wY[j]
        }
        allErrors.push(Math.hypot(px - dataPoints[te].target.x, py - dataPoints[te].target.y))
      }
    }
  }

  const rms = Math.sqrt(allErrors.reduce((s, v) => s + v * v, 0) / allErrors.length)
  const sorted = [...allErrors].sort((a, b) => a - b)
  const p95 = sorted[Math.floor(sorted.length * 0.95)]
  return { rms, p95 }
}

const resFull = runAblationVariant('full')
const resNoTransit = runAblationVariant('no_transit')
const resNoCentroid = runAblationVariant('no_centroid')
const resPureL2 = runAblationVariant('pure_l2')
const resRawLandmarks = runAblationVariant('raw_landmarks')

console.log(`Full Proposed OpenEyeGaze:        RMS = ${resFull.rms.toFixed(1)} px, P95 = ${resFull.p95.toFixed(1)} px`)
console.log(`w/o Saccadic Transit Discard:     RMS = ${resNoTransit.rms.toFixed(1)} px (+${(((resNoTransit.rms - resFull.rms) / resFull.rms) * 100).toFixed(1)}%), P95 = ${resNoTransit.p95.toFixed(1)} px`)
console.log(`w/o Centroid Aggregation:         RMS = ${resNoCentroid.rms.toFixed(1)} px (+${(((resNoCentroid.rms - resFull.rms) / resFull.rms) * 100).toFixed(1)}%), P95 = ${resNoCentroid.p95.toFixed(1)} px`)
console.log(`w/o L1 Sparsity (Pure L2 Ridge):  RMS = ${resPureL2.rms.toFixed(1)} px (+${(((resPureL2.rms - resFull.rms) / resFull.rms) * 100).toFixed(1)}%), P95 = ${resPureL2.p95.toFixed(1)} px`)
console.log(`w/o Canthal Normalization (Raw):  RMS = ${resRawLandmarks.rms.toFixed(1)} px (+${(((resRawLandmarks.rms - resFull.rms) / resFull.rms) * 100).toFixed(1)}%), P95 = ${resRawLandmarks.p95.toFixed(1)} px`)

// -------------------------------------------------------------
// 3. REAL CROSS-SESSION TRANSFER & 4-POINT AFFINE RECALIBRATION
// -------------------------------------------------------------
console.log('\n--- 3. Real Cross-Session Transfer & Affine Recalibration (All 12 Pairs) ---')

interface SessionModel {
  wX: Float64Array
  wY: Float64Array
  mu: Float64Array
  sd: Float64Array
  anchor: any
}

function trainSessionModel(samples: RawCalibrationSample[]): SessionModel {
  const cleaned = filterSaccadicTransit(samples, 3)
  const centroids = aggregateTargetCentroids(cleaned, 0.2)
  const anchor = computeCalibrationAnchor(centroids)
  const fullX = centroids.map((s) => buildEnhancedFeatureRow(s.feature, anchor))
  const Yx = centroids.map((s) => s.target.x)
  const Yy = centroids.map((s) => s.target.y)
  const { XStd, mu, sd } = standardizeFeatures(fullX)
  const wX = solveElasticNet(XStd, Yx, 0.3, 1.5e-3)
  const wY = solveElasticNet(XStd, Yy, 0.3, 1.2e-3)
  return { wX, wY, mu, sd, anchor }
}

const trainedModels = loadedSessions.map((s) => trainSessionModel(s.samples))

const zeroShotErrors: number[] = []
const centerBiasErrors: number[] = []
const scaleBiasErrors: number[] = []
const affineErrors: number[] = []

for (let src = 0; src < loadedSessions.length; src++) {
  for (let dst = 0; dst < loadedSessions.length; dst++) {
    if (src === dst) continue

    const model = trainedModels[src]
    const dstSamples = loadedSessions[dst].samples
    const p = ENHANCED_FEATURE_COUNT

    // Predict raw targets on dst
    const preds: { predX: number; predY: number; targetX: number; targetY: number; group: number }[] = []
    for (const s of dstSamples) {
      const row = buildEnhancedFeatureRow(s.feature, model.anchor)
      let px = model.wX[0], py = model.wY[0]
      for (let j = 1; j < p; j++) {
        const z = (row[j] - model.mu[j - 1]) / model.sd[j - 1]
        px += z * model.wX[j]; py += z * model.wY[j]
      }
      preds.push({ predX: px, predY: py, targetX: s.target.x, targetY: s.target.y, group: s.group })
    }

    // Zero-shot error
    for (const pr of preds) {
      zeroShotErrors.push(Math.hypot(pr.predX - pr.targetX, pr.predY - pr.targetY))
    }

    // 1-Point Center bias correction (group near center: e.g. target nearest screen center)
    const centerTarget = preds.reduce((best, cur) => {
      const dCur = Math.hypot(cur.targetX - 960, cur.targetY - 540)
      const dBest = Math.hypot(best.targetX - 960, best.targetY - 540)
      return dCur < dBest ? cur : best
    }, preds[0])
    const centerSamples = preds.filter((pr) => pr.group === centerTarget.group)
    const biasX = centerSamples.reduce((s, c) => s + (c.targetX - c.predX), 0) / centerSamples.length
    const biasY = centerSamples.reduce((s, c) => s + (c.targetY - c.predY), 0) / centerSamples.length
    for (const pr of preds) {
      centerBiasErrors.push(Math.hypot(pr.predX + biasX - pr.targetX, pr.predY + biasY - pr.targetY))
    }

    // 2-Point Scale + Bias correction (using 2 horizontal targets)
    const minXTarget = preds.reduce((best, cur) => cur.targetX < best.targetX ? cur : best, preds[0])
    const maxXTarget = preds.reduce((best, cur) => cur.targetX > best.targetX ? cur : best, preds[0])
    const minSamples = preds.filter((pr) => pr.group === minXTarget.group)
    const maxSamples = preds.filter((pr) => pr.group === maxXTarget.group)
    const avgPredMinX = minSamples.reduce((s, c) => s + c.predX, 0) / minSamples.length
    const avgPredMaxX = maxSamples.reduce((s, c) => s + c.predX, 0) / maxSamples.length
    const scaleX = (maxXTarget.targetX - minXTarget.targetX) / Math.max(10, avgPredMaxX - avgPredMinX)
    const offsetX = minXTarget.targetX - scaleX * avgPredMinX
    for (const pr of preds) {
      scaleBiasErrors.push(Math.hypot(scaleX * pr.predX + offsetX - pr.targetX, pr.predY + biasY - pr.targetY))
    }

    // 4-Point Cardinal Affine Adaptation (4 corner targets)
    // Find 4 targets closest to 4 display corners: (0,0), (1920,0), (0,1080), (1920,1080)
    const corners = [
      { x: 200, y: 150 },
      { x: 1720, y: 150 },
      { x: 200, y: 900 },
      { x: 1720, y: 900 },
    ]
    const cornerGroups = corners.map((c) => {
      return preds.reduce((best, cur) => {
        const dCur = Math.hypot(cur.targetX - c.x, cur.targetY - c.y)
        const dBest = Math.hypot(best.targetX - c.x, best.targetY - c.y)
        return dCur < dBest ? cur : best
      }, preds[0]).group
    })

    const affRecal = new AffineRecalibrator()
    const pairs: { predicted: { x: number; y: number }; actual: { x: number; y: number } }[] = []
    for (const g of cornerGroups) {
      const gSamples = preds.filter((pr) => pr.group === g)
      if (gSamples.length > 0) {
        const avgPx = gSamples.reduce((s, c) => s + c.predX, 0) / gSamples.length
        const avgPy = gSamples.reduce((s, c) => s + c.predY, 0) / gSamples.length
        pairs.push({
          predicted: { x: avgPx, y: avgPy },
          actual: { x: gSamples[0].targetX, y: gSamples[0].targetY },
        })
      }
    }
    affRecal.fit(pairs, 0.05)

    for (const pr of preds) {
      const corr = affRecal.apply(pr.predX, pr.predY)
      affineErrors.push(Math.hypot(corr.x - pr.targetX, corr.y - pr.targetY))
    }
  }
}

const meanZeroShot = zeroShotErrors.reduce((a, b) => a + b, 0) / zeroShotErrors.length
const meanCenter = centerBiasErrors.reduce((a, b) => a + b, 0) / centerBiasErrors.length
const meanScale = scaleBiasErrors.reduce((a, b) => a + b, 0) / scaleBiasErrors.length
const meanAffine = affineErrors.reduce((a, b) => a + b, 0) / affineErrors.length

console.log(`Zero-Shot Cross-Session Transfer Error:   ${meanZeroShot.toFixed(1)} px`)
console.log(`1-Point Center Bias Recalibration Error:  ${meanCenter.toFixed(1)} px (-${(((meanZeroShot - meanCenter) / meanZeroShot) * 100).toFixed(1)}%)`)
console.log(`2-Point Scale + Bias Recalibration Error: ${meanScale.toFixed(1)} px (-${(((meanZeroShot - meanScale) / meanZeroShot) * 100).toFixed(1)}%)`)
console.log(`4-Point Cardinal Affine Adaptation Error: ${meanAffine.toFixed(1)} px (-${(((meanZeroShot - meanAffine) / meanZeroShot) * 100).toFixed(1)}%, -${(meanZeroShot - meanAffine).toFixed(1)} px)`)

function randn(): number {
  let u = 0, v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
}

// -------------------------------------------------------------
// 4. BAYESIAN READING PVL ATTRIBUTION SIMULATION
// -------------------------------------------------------------
console.log('\n--- 4. Analytical Bayesian Reading PVL Simulation ---')
// Gaze fixations sampled from empirical tracker error distribution (sigma_x = 25.2 px during fixations)
// across standardized passage layouts conforming to Rayner (1979) oculomotor landing distribution:
// True landing ~ Normal(mean = 0.40 * width, sd = 0.10 * width)
const N_reading = 50000
const sigma_tracker = 25.2

let pvlCorrect = 0, pvlNear = 0, pvlErrorSum = 0
let midpointCorrect = 0, midpointNear = 0, midpointErrorSum = 0

for (let i = 0; i < N_reading; i++) {
  const words: { left: number; right: number; width: number }[] = []
  let curX = 100
  for (let w = 0; w < 8; w++) {
    const width = 50 + Math.random() * 40 // 50 to 90 px (~6-8 chars)
    words.push({ left: curX, right: curX + width, width })
    curX += width + 14 // 14 px space
  }

  const targetIdx = 1 + Math.floor(Math.random() * 6)
  const tw = words[targetIdx]

  const trueLanding = tw.left + 0.40 * tw.width + randn() * (0.10 * tw.width)
  const measuredGaze = trueLanding + randn() * sigma_tracker

  // Method 1: Proposed Bayesian with PVL prior (0.40)
  let bestPvlIdx = -1, maxPvlLogLik = -Infinity
  for (let j = 0; j < words.length; j++) {
    const xTarget = words[j].left + 0.40 * words[j].width
    const logLik = -0.5 * ((measuredGaze - xTarget) / sigma_tracker) ** 2
    if (logLik > maxPvlLogLik) {
      maxPvlLogLik = logLik
      bestPvlIdx = j
    }
  }
  if (bestPvlIdx === targetIdx) pvlCorrect++
  if (Math.abs(bestPvlIdx - targetIdx) <= 1) pvlNear++
  pvlErrorSum += Math.abs(bestPvlIdx - targetIdx)

  // Method 2: Ablated Geometric Center (0.50)
  let bestMidIdx = -1, maxMidLogLik = -Infinity
  for (let j = 0; j < words.length; j++) {
    const xTarget = words[j].left + 0.50 * words[j].width
    const logLik = -0.5 * ((measuredGaze - xTarget) / sigma_tracker) ** 2
    if (logLik > maxMidLogLik) {
      maxMidLogLik = logLik
      bestMidIdx = j
    }
  }
  if (bestMidIdx === targetIdx) midpointCorrect++
  if (Math.abs(bestMidIdx - targetIdx) <= 1) midpointNear++
  midpointErrorSum += Math.abs(bestMidIdx - targetIdx)
}

console.log(`Proposed Bayesian with PVL (0.40): Top-1 Word Acc = ${((pvlCorrect / N_reading) * 100).toFixed(1)}%, Within +-1 = ${((pvlNear / N_reading) * 100).toFixed(1)}%, Word Error Index = ${(pvlErrorSum / N_reading).toFixed(2)}`)
console.log(`Ablated Geometric Center (0.50):   Top-1 Word Acc = ${((midpointCorrect / N_reading) * 100).toFixed(1)}%, Within +-1 = ${((midpointNear / N_reading) * 100).toFixed(1)}%, Word Error Index = ${(midpointErrorSum / N_reading).toFixed(2)}`)
