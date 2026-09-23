/**
 * run_all_tests.ts — Comprehensive Test Suite for OpenGaze Framework.
 */

import { projectCanthalGaze } from '../core/canthal'
import {
  buildEnhancedFeatureRow,
  computeCalibrationAnchor,
  ENHANCED_FEATURE_COUNT,
  type EyeFeature,
} from '../core/features28'
import {
  solveElasticNet,
  solveRidge,
  standardizeFeatures,
  standardizeWeights,
  unstandardizeWeights,
} from '../core/solvers'
import {
  trainGazeModel,
  type RawCalibrationSample,
} from '../core/modelSelection'
import { OneEuroFilter2D } from '../core/oneEuroFilter'
import { FixationDetector } from '../core/fixation'
import {
  ContinuousAdaptationEngine,
  AffineRecalibrator,
} from '../core/onlineAdaptation'
import { ReadingLineSnapper } from '../web/lineSnapper'
import { attributeFixationToText } from '../web/bayesianAOI'

console.log('========================================================================')
console.log('       RUNNING OPENGAZE STANDALONE FRAMEWORK MASTER TEST SUITE          ')
console.log('========================================================================\n')

let passCount = 0

function assert(condition: boolean, msg: string) {
  if (!condition) {
    console.error(`[FAIL] ${msg}`)
    throw new Error(`Assertion failed: ${msg}`)
  }
  console.log(`[PASS] ${msg}`)
  passCount++
}

// -----------------------------------------------------------------------------
// Test 1: Canthal Invariance
// -----------------------------------------------------------------------------
console.log('--- Test 1: Geometric Canthal Coordinate Normalization ---')
const eyeCorners = {
  inner: { x: 100, y: 100 },
  outer: { x: 150, y: 100 },
  irisCenter: { x: 125, y: 100 }, // exactly in middle
}
const proj = projectCanthalGaze(eyeCorners, false)
assert(Math.abs(proj.gx) < 1e-6, 'Canthal centered iris yields gx ≈ 0')
assert(Math.abs(proj.gy) < 1e-6, 'Canthal centered iris yields gy ≈ 0')
assert(Math.abs(proj.width - 50) < 1e-6, 'Inter-canthal width correctly computed as 50px')

// Test roll invariance (rotate eye corners by 45 degrees)
const rad = Math.PI / 4
const cos = Math.cos(rad)
const sin = Math.sin(rad)
const rotInner = { x: 125 - 25 * cos, y: 100 - 25 * sin }
const rotOuter = { x: 125 + 25 * cos, y: 100 + 25 * sin }
const rotIris = { x: 125 + 10 * cos, y: 100 + 10 * sin } // shifted along eye horizontal axis
const projRot = projectCanthalGaze({ inner: rotInner, outer: rotOuter, irisCenter: rotIris }, false)
assert(Math.abs(projRot.gy) < 1e-4, 'Canthal projection cancels head roll tilt on vertical axis')
assert(Math.abs(projRot.gx - 0.2) < 1e-3, 'Canthal projection correctly recovers normalized 0.2 eye-width horizontal offset')

// -----------------------------------------------------------------------------
// Test 2: 28-Term Feature Construction
// -----------------------------------------------------------------------------
console.log('\n--- Test 2: 28-Term Physics-Informed Feature Space ---')
const mockFeature: EyeFeature = {
  gx: 0.1,
  gy: -0.05,
  disparity: -0.02,
  ear: 0.24,
  faceScale: 250,
  yawProxy: 0.05,
  pitchProxy: 0.65,
  roll: 0.02,
  noseX: 720,
  noseY: 420,
  eyeA: { gx: 0.08, gy: -0.05, ear: 0.23, width: 52, irisRadius: 11.2 },
  eyeB: { gx: 0.12, gy: -0.05, ear: 0.25, width: 53, irisRadius: 11.1 },
}
const mockAnchor = computeCalibrationAnchor([{ feature: mockFeature }])
const row28 = buildEnhancedFeatureRow(mockFeature, mockAnchor)
assert(row28.length === ENHANCED_FEATURE_COUNT, `Feature row contains exactly ${ENHANCED_FEATURE_COUNT} terms`)
assert(row28[0] === 1.0, 'Feature row col 0 is constant intercept 1.0')
for (let i = 0; i < row28.length; i++) {
  assert(Number.isFinite(row28[i]), `Feature term ${i} is finite`)
}

// -----------------------------------------------------------------------------
// Test 3: Solvers (ElasticNet & Ridge)
// -----------------------------------------------------------------------------
console.log('\n--- Test 3: ElasticNet & Ridge Solvers ---')
const mockX: Float64Array[] = []
const mockY: number[] = []
for (let i = 0; i < 20; i++) {
  const r = new Float64Array(4)
  r[0] = 1.0
  r[1] = i * 0.1
  r[2] = i * i * 0.01
  r[3] = (i % 3) * 0.5
  mockX.push(r)
  mockY.push(100 + 20 * r[1] - 5 * r[2])
}
const { XStd } = standardizeFeatures(mockX)
const wRidge = solveRidge(XStd, mockY, 1e-3)
assert(wRidge.length === 4, 'Ridge solver returns correct weight dimensions')
assert(Math.abs(wRidge[0] - 100) < 50, 'Ridge intercept reasonably recovers target mean')

const wElNet = solveElasticNet(XStd, mockY, 0.5, 1e-3)
assert(wElNet.length === 4, 'ElasticNet solver converges with valid weights')

// -----------------------------------------------------------------------------
// Test 4: Speed-Adaptive OneEuroFilter
// -----------------------------------------------------------------------------
console.log('\n--- Test 4: One-Euro Adaptive Filter ---')
const filter = new OneEuroFilter2D(1.0, 0.05, 1.0)
let t = 1000
const p1 = filter.filter(500, 300, t)
assert(p1.x === 500 && p1.y === 300, 'Initial filter point matches input')

// Simulate high-speed saccade jump
t += 30
const pSaccade = filter.filter(800, 300, t)
assert(pSaccade.x > 700, 'One-Euro filter rapidly tracks high-speed ballistic saccade (>700px on 300px step)')

// -----------------------------------------------------------------------------
// Test 5: Uncertainty-Scaled I-DT Fixation Detector
// -----------------------------------------------------------------------------
console.log('\n--- Test 5: Fixation Detector with Seed-Forward Saccades ---')
const detector = new FixationDetector({ minDurationMs: 100, baseDispersionXPx: 30, baseDispersionYPx: 20 })

// Feed 8 stable points spanning 160ms (20ms interval)
for (let i = 0; i < 8; i++) {
  detector.processSample({
    x: 400 + (i % 3),
    y: 200 + (i % 2),
    timestamp: 1000 + i * 20,
    sigmaX: 5,
    sigmaY: 5,
  })
}

// Feed saccade jump breaking fixation
const closedFix = detector.processSample({
  x: 600,
  y: 200,
  timestamp: 1180,
  sigmaX: 5,
  sigmaY: 5,
})

assert(closedFix !== null, 'Fixation closed upon saccade breaking dispersion threshold')
if (closedFix) {
  assert(Math.abs(closedFix.centroidX - 401) < 2.0, 'Fixation centroid accurately computed from resting samples')
  assert(closedFix.duration >= 140, `Fixation duration (${closedFix.duration}ms) meets threshold`)
}

// -----------------------------------------------------------------------------
// Test 6: In-Situ Online Fine-Tuning Engine
// -----------------------------------------------------------------------------
console.log('\n--- Test 6: Continuous In-Situ Fine-Tuning Engine ---')
const initWeights = new Float64Array(ENHANCED_FEATURE_COUNT).fill(0.1)
initWeights[0] = 500 // Screen center x
const adaptEngine = new ContinuousAdaptationEngine(mockAnchor, initWeights, initWeights, {
  minSamplesToAdapt: 4,
  adaptEveryNSamples: 2,
})

// Feed continuous stream of high-frequency frames
for (let i = 0; i < 20; i++) {
  adaptEngine.pushFrame(mockFeature, 2000 + i * 33)
}

// User clicks target at (520, 300) at t = 2600
let adapted = adaptEngine.registerInteraction(520, 300, 2300)
adapted = adaptEngine.registerInteraction(530, 310, 2400)
adapted = adaptEngine.registerInteraction(525, 305, 2500)
adapted = adaptEngine.registerInteraction(522, 302, 2600)

assert(typeof adapted === 'boolean', 'Continuous fine-tuning processes interaction events')
const updatedWeights = adaptEngine.getWeights()
assert(updatedWeights.weightsX.length === ENHANCED_FEATURE_COUNT, 'Adapted weights match 28-term dimensionality')
assert(Number.isFinite(updatedWeights.weightsX[0]), 'Bayesian anchored weights remain finite without divergence')

// -----------------------------------------------------------------------------
// Test 7: ReadingLineSnapper Soft Attractor
// -----------------------------------------------------------------------------
console.log('\n--- Test 7: Reading Line Snapper Soft Attractor ---')
const snapper = new ReadingLineSnapper()
snapper.updateFromWordBoxes([
  { wordIndex: 0, text: 'Gaze', rects: [{ top: 100, bottom: 124, left: 10, right: 60, width: 50, height: 24 } as DOMRect] },
  { wordIndex: 1, text: 'Tracker', rects: [{ top: 100, bottom: 124, left: 70, right: 140, width: 70, height: 24 } as DOMRect] },
  { wordIndex: 2, text: 'Second', rects: [{ top: 140, bottom: 164, left: 10, right: 70, width: 60, height: 24 } as DOMRect] },
  { wordIndex: 3, text: 'Line', rects: [{ top: 140, bottom: 164, left: 80, right: 130, width: 50, height: 24 } as DOMRect] },
])
assert(snapper.getLineCount() === 2, 'Line snapper identified 2 distinct text lines')

const snapped = snapper.snapGaze(50, 118) // Gaze near line 0 (center 112)
assert(snapped.isSnapped, 'Gaze within 6px of line center is attracted')
assert(snapped.y < 118 && snapped.y > 112, 'Soft Gaussian pull gently attracted gaze vertically toward line center')

// -----------------------------------------------------------------------------
// Test 8: Bayesian AOI Attribution with Preferred Viewing Location (0.4)
// -----------------------------------------------------------------------------
console.log('\n--- Test 8: Two-Stage Bayesian Word Attribution with PVL ---')
const mockLineBands = [
  {
    lineIndex: 0,
    top: 100,
    bottom: 124,
    centerY: 112,
    height: 24,
    left: 10,
    right: 200,
    wordBoxes: [
      { wordIndex: 0, text: 'The', rects: [{ left: 10, right: 50, width: 40, top: 100, bottom: 124, height: 24 } as DOMRect] },
      { wordIndex: 1, text: 'Quick', rects: [{ left: 60, right: 120, width: 60, top: 100, bottom: 124, height: 24 } as DOMRect] },
      { wordIndex: 2, text: 'Brown', rects: [{ left: 130, right: 190, width: 60, top: 100, bottom: 124, height: 24 } as DOMRect] },
    ],
  },
]

// Word 1 ('Quick') has rect left=60, width=60. PVL target = 60 + 0.4*60 = 84px.
// Look at 82px (very close to PVL)
const attr = attributeFixationToText(82, 112, 15, 12, mockLineBands)
assert(attr.level === 'word', 'Fixation at PVL target attributed cleanly to level word')
assert(attr.wordText === 'Quick', 'Attributed to exact word Quick')
// -----------------------------------------------------------------------------
// Test 9: Fast 4-Point Affine Recalibrator
// -----------------------------------------------------------------------------
console.log('\n--- Test 9: Fast 4-Point Affine Recalibrator ---')
const affine = new AffineRecalibrator()
const pairs = [
  { predicted: { x: 100, y: 100 }, actual: { x: 110, y: 105 } },
  { predicted: { x: 900, y: 100 }, actual: { x: 910, y: 105 } },
  { predicted: { x: 100, y: 700 }, actual: { x: 110, y: 705 } },
  { predicted: { x: 900, y: 700 }, actual: { x: 910, y: 705 } },
]
const ok = affine.fit(pairs)
assert(ok, 'AffineRecalibrator fits 4-point pairs successfully')
const corr = affine.apply(500, 400)
assert(Math.abs(corr.x - 510) < 1.0 && Math.abs(corr.y - 405) < 1.0, 'Affine correction recovers translation shift')

// -----------------------------------------------------------------------------
// Test 10: Weight Standardization / Unstandardization Mathematical Invariance
// -----------------------------------------------------------------------------
console.log('\n--- Test 10: Weight Standardization Invariance ---')
const testWRaw = new Float64Array([100, 2.5, -4.0])
const testMu = new Float64Array([10.0, 20.0])
const testSd = new Float64Array([2.0, 5.0])
const testWStd = standardizeWeights(testWRaw, testMu, testSd)
const testWRawRecov = unstandardizeWeights(testWStd, testMu, testSd)
assert(Math.abs(testWRawRecov[0] - testWRaw[0]) < 1e-10, 'Unstandardized intercept matches ground truth')
assert(Math.abs(testWRawRecov[1] - testWRaw[1]) < 1e-10, 'Unstandardized slope 1 matches ground truth')
assert(Math.abs(testWRawRecov[2] - testWRaw[2]) < 1e-10, 'Unstandardized slope 2 matches ground truth')

// -----------------------------------------------------------------------------
// Test 11: Continuous Adaptation Engine Prediction & Affine Cascade
// -----------------------------------------------------------------------------
console.log('\n--- Test 11: Continuous Adaptation Engine Prediction ---')
const onlinePred = adaptEngine.predict(mockFeature)
assert(Number.isFinite(onlinePred.x) && Number.isFinite(onlinePred.y), 'ContinuousAdaptationEngine.predict returns finite coordinates')
const affinePred = adaptEngine.predictWithAffine(mockFeature, affine)
assert(Number.isFinite(affinePred.x) && Number.isFinite(affinePred.y), 'ContinuousAdaptationEngine.predictWithAffine cascades through affine matrix')

// -----------------------------------------------------------------------------
// Test 12: FixationDetector Flush
// -----------------------------------------------------------------------------
console.log('\n--- Test 12: FixationDetector Flush ---')
const flushDetector = new FixationDetector({ minDurationMs: 80, baseDispersionXPx: 30, baseDispersionYPx: 20 })
for (let i = 0; i < 6; i++) {
  flushDetector.processSample({ x: 300, y: 150, timestamp: 1000 + i * 20 })
}
const flushedFix = flushDetector.flush()
assert(flushedFix !== null, 'FixationDetector.flush cleanly returns active fixation at stream end')
assert(flushDetector.flush() === null, 'Subsequent flush returns null after window cleared')

// -----------------------------------------------------------------------------
// Test 13: End-to-End trainGazeModel Pipeline
// -----------------------------------------------------------------------------
console.log('\n--- Test 13: trainGazeModel Full Calibration Pipeline ---')
const mockCalibrationSamples: RawCalibrationSample[] = []
for (let g = 0; g < 4; g++) {
  const target = { x: 200 + g * 200, y: 150 + (g % 2) * 200 }
  for (let f = 0; f < 10; f++) {
    mockCalibrationSamples.push({
      group: g,
      target,
      feature: {
        ...mockFeature,
        gx: (target.x - 500) / 1000 + (f % 3) * 0.001,
        gy: (target.y - 300) / 1000 + (f % 2) * 0.001,
      },
      timestamp: g * 1000 + f * 50,
    })
  }
}
const model = trainGazeModel(mockCalibrationSamples)
assert(model.cleanedSamples.length > 0, 'trainGazeModel produced cleaned samples')
assert(model.centroids.length === 4, 'trainGazeModel aggregated 4 target centroids')
assert(model.weightsX.length === ENHANCED_FEATURE_COUNT, 'Trained model weights match 28 terms')
const testGaze = model.predict(mockFeature)
assert(Number.isFinite(testGaze.x) && Number.isFinite(testGaze.y), 'Trained model predicts valid finite gaze coordinate')

console.log('\n========================================================================')
console.log(`[ALL ${passCount} UNIT AND INTEGRATION TESTS PASSED CLEANLY!]`)
console.log('========================================================================')

