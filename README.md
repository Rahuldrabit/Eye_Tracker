# OpenEyeGaze: Head-Invariant In-Browser Eye-Tracking Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue.svg)](https://www.typescriptlang.org/)
[![Repository](https://img.shields.io/badge/GitHub-Rahuldrabit%2FEye__Tracker-green.svg)](https://github.com/Rahuldrabit/Eye_Tracker)

**OpenEyeGaze** is a research-grade, zero-DOM web eye-tracking framework engineered for commodity webcams. It solves the classic failure modes of browser eye tracking—**head-sway vulnerability**, **ill-conditioned polynomial calibration**, **temporal posture drift**, and **DOM-bloating markup**—through closed-form canthal geometry, multi-task ElasticNet optimization, and in-situ online fine-tuning.

---

## 🔬 Key Scientific Innovations

1. **Head-Invariant Canthal Vector Normalization**:
   - Iris centers are projected onto skull-anchored inter-canthal unit vectors $(\mathbf{u}, \mathbf{v})$ and normalized by inter-canthal width $w$.
   - Cancels 1st-order lateral head translation, distance changes, and head roll tilt without requiring dense 3D facial meshes.
2. **28-Term Physics-Informed Feature Space & ElasticNet**:
   - Accounts for perspective foreshortening on iris radius, biological depth ($Z_{\text{ratio}}$), binocular disparity, eyelid occlusion, and cubic screen-tangent projections.
   - Solved via cyclical coordinate descent with dual-axis independent regularization ($\alpha_x, \alpha_y$).
3. **In-Situ Online Fine-Tuning Engine (`ContinuousAdaptationEngine`)**:
   - **Continuous passive adaptation**: Harvests supervised $(feature, target)$ pairs from natural user interactions (clicks, taps) using human eye-hand foveation coupling ($\sim 80\text{--}240\,$ms lead time).
   - **Bayesian Prior Anchor ($W_0$)**: Solves with an elastic penalty tying the model to initial calibration weights:
     $$\min_W \|Y - X W\|^2 + \lambda \|W\|^2 + \gamma \|W - W_0\|^2$$
     Guarantees the tracker improves continuously during usage without diverging or distorting unvisited screen quadrants.
4. **Zero-DOM Bayesian Reading Telemetry**:
   - Uses `Intl.Segmenter` and `Range.getClientRects()` to index raw text nodes directly from browser layout trees. **Zero `<span>` wrappers needed**.
   - Incorporates human **Preferred Viewing Location (PVL = 0.4)** to eliminate systematic rightward word-attribution bias.
5. **Gate-Before-Filter One-Euro Temporal Smoothing**:
   - EAR blink rejection and Median Absolute Deviation (MAD) outlier pruning happen *before* the speed-adaptive One-Euro filter (Casiez et al., CHI 2012), preventing rejected frames from corrupting internal filter states.

---

## 📊 Empirical Baseline Benchmark (ACM ETRA 2027 Evaluation)

Evaluated across $N=3$ human participants (4 sessions, 1,693 captured frames) using held-out **Grouped Leave-One-Target-Out Cross-Validation (20-Fold)**:

| Model Architecture | CV RMS (px) $\downarrow$ | Visual Angle ($^\circ$) $\downarrow$ | P95 Error (px) $\downarrow$ | Cond. $\kappa(A^T A)$ $\downarrow$ | Live Inference |
|---|:---:|:---:|:---:|:---:|:---:|
| Classical Polynomial (WebGazer OLS) | 211.5 | $5.53^\circ$ | 399.5 | 34.3 | 0.008 ms |
| Standard Linear Ridge | 220.2 | $5.76^\circ$ | 411.6 | 872.0 | 0.005 ms |
| Standard Polynomial Ridge | 211.6 | $5.53^\circ$ | 398.3 | $5.4 \times 10^5$ | 0.011 ms |
| Production Per-Eye Ridge Baseline | 221.0 | $5.78^\circ$ | 405.3 | 900.1 | 0.008 ms |
| **OpenEyeGaze (Proposed 28-Term)** | **123.1** | **3.22$^\circ$** | **254.3** | $5.0 \times 10^{10}$ | **0.020 ms** |
| *With Multivariate Outlier Rejection* | **111.1** | **2.91$^\circ$** | **227.6** | --- | **0.020 ms** |

> *Note: Live single-frame inference latency is **0.02 ms** (20 microseconds, consuming <0.15% of a 60 FPS frame). Full 20-fold cross-validation calibration takes only 0.83 ms total on a single CPU thread.*

---

## 🚀 Quick Start

### Installation

```bash
npm install @rahuldrabit/eye-tracker
```

### 1. Basic Tracking Pipeline

```typescript
import {
  CameraManager,
  OneEuroFilter2D,
  FixationDetector,
  projectCanthalGaze
} from '@rahuldrabit/eye-tracker'

const camera = new CameraManager()
const filter = new OneEuroFilter2D(1.0, 0.05, 1.0)
const fixationDetector = new FixationDetector()

await camera.start(videoElement, (video, timestampMs) => {
  // 1. Extract eye landmarks from MediaPipe FaceMesh
  // 2. Project onto canthal basis
  const { gx, gy } = projectCanthalGaze(leftEyeCorners, true)

  // 3. Smooth with speed-adaptive One-Euro filter
  const smoothed = filter.filter(rawScreenX, rawScreenY, timestampMs)

  // 4. Detect fixations and saccades
  const fixation = fixationDetector.processSample({
    x: smoothed.x,
    y: smoothed.y,
    timestamp: timestampMs
  })

  if (fixation) {
    console.log(`Fixation detected at (${fixation.centroidX}, ${fixation.centroidY}) duration: ${fixation.duration}ms`)
  }
})
```

### 2. In-Situ Continuous Fine-Tuning (Self-Improving Tracker)

```typescript
import { ContinuousAdaptationEngine } from '@rahuldrabit/eye-tracker'

const adaptationEngine = new ContinuousAdaptationEngine(
  calibrationAnchor,
  initialWeightsX,
  initialWeightsY,
  {
    minSamplesToAdapt: 6,
    adaptEveryNSamples: 3,
    anchorWeightGamma: 5.0, // Elastic anchor to W_0 prevents divergence
  }
)

// In your camera frame loop:
adaptationEngine.pushFrame(currentEyeFeature, timestampMs)

// On any window click event:
window.addEventListener('click', (event) => {
  const didAdapt = adaptationEngine.registerInteraction(event.clientX, event.clientY, performance.now())
  if (didAdapt) {
    const { weightsX, weightsY } = adaptationEngine.getWeights()
    console.log('Model smoothly adapted to user posture shift!')
  }
})
```

### 3. Zero-DOM Reading Attribution

```typescript
import {
  extractWordBoxesFromElement,
  ReadingLineSnapper,
  attributeFixationToText
} from '@rahuldrabit/eye-tracker'

// 1. Measure DOM text boxes without modifying HTML
const container = document.getElementById('reading-pane')!
const wordBoxes = extractWordBoxesFromElement(container)

// 2. Setup line snapper attractor
const snapper = new ReadingLineSnapper()
snapper.updateFromWordBoxes(wordBoxes)

// 3. Snap gaze and attribute with PVL (0.4)
const snappedGaze = snapper.snapGaze(rawGazeX, rawGazeY)
const attribution = attributeFixationToText(snappedGaze.x, snappedGaze.y, sigmaX, sigmaY, snapper.lines)

if (attribution.level === 'word') {
  console.log(`Reading word: "${attribution.wordText}" (Confidence: ${(attribution.confidence * 100).toFixed(1)}%)`)
}
```

---

## 🧪 Running Benchmarks & Tests

```bash
# Run complete unit and integration test suite (52 tests)
npm test

# Run empirical ACM ETRA benchmark suite across calibration datasets
npm run benchmark
```

---

## 📄 Academic Citation (ACM ETRA 2027)

```bibtex
@inproceedings{drabit2027openeyegaze,
  title={OpenEyeGaze: Head-Invariant In-Browser Gaze Estimation with In-Situ Online Fine-Tuning and Zero-DOM Bayesian Reading Telemetry},
  author={Drabit, Rahul},
  booktitle={Proceedings of the 2027 Symposium on Eye Tracking Research and Applications (ETRA '27)},
  year={2027},
  publisher={ACM}
}
```

---

## ⚖️ License

MIT License. Designed & Developed by **Rahul Drabit**.
