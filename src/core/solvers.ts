/**
 * solvers.ts — Regularized Regression Solvers for Gaze Estimation.
 *
 * Implements:
 * 1. Multi-task ElasticNet Coordinate Descent (L1 Sparsity + L2 Ridge).
 * 2. Tikhonov Ridge Regression with predictor standardization.
 * 3. Numerical stability routines (Gaussian elimination with partial pivoting).
 */

export interface StandardizationStats {
  mu: Float64Array
  sd: Float64Array
}

export function standardizeFeatures(X: Float64Array[]): {
  XStd: Float64Array[]
  mu: Float64Array
  sd: Float64Array
} {
  const n = X.length
  if (n === 0) {
    return {
      XStd: [],
      mu: new Float64Array(0),
      sd: new Float64Array(0),
    }
  }
  const p = X[0].length
  const mu = new Float64Array(p - 1)
  const sd = new Float64Array(p - 1)

  for (let j = 1; j < p; j++) {
    let sum = 0
    for (let i = 0; i < n; i++) sum += X[i][j]
    mu[j - 1] = sum / n

    let sumSq = 0
    for (let i = 0; i < n; i++) {
      const diff = X[i][j] - mu[j - 1]
      sumSq += diff * diff
    }
    const std = Math.sqrt(sumSq / n)
    sd[j - 1] = std < 1e-9 ? 1.0 : std
  }

  const XStd: Float64Array[] = []
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(p)
    row[0] = 1.0 // Intercept remains 1.0
    for (let j = 1; j < p; j++) {
      row[j] = (X[i][j] - mu[j - 1]) / sd[j - 1]
    }
    XStd.push(row)
  }

  return { XStd, mu, sd }
}

/**
 * Maps standardized regression weights back to the original unstandardized feature space:
 *   w_raw[j] = w_std[j] / sd[j-1]  (for j >= 1)
 *   w_raw[0] = w_std[0] - sum_{j=1}^{p-1} (w_std[j] * mu[j-1] / sd[j-1])
 */
export function unstandardizeWeights(
  wStd: Float64Array,
  mu: Float64Array,
  sd: Float64Array
): Float64Array {
  const p = wStd.length
  const wRaw = new Float64Array(p)
  let intercept = wStd[0]
  for (let j = 1; j < p; j++) {
    const s = sd[j - 1] > 1e-9 ? sd[j - 1] : 1.0
    wRaw[j] = wStd[j] / s
    intercept -= (wStd[j] * mu[j - 1]) / s
  }
  wRaw[0] = intercept
  return wRaw
}

/**
 * Maps unstandardized raw weights into standardized predictor space for a given mu and sd:
 *   w_std[j] = w_raw[j] * sd[j-1]  (for j >= 1)
 *   w_std[0] = w_raw[0] + sum_{j=1}^{p-1} (w_raw[j] * mu[j-1])
 */
export function standardizeWeights(
  wRaw: Float64Array,
  mu: Float64Array,
  sd: Float64Array
): Float64Array {
  const p = wRaw.length
  const wStd = new Float64Array(p)
  let intercept = wRaw[0]
  for (let j = 1; j < p; j++) {
    const s = sd[j - 1] > 1e-9 ? sd[j - 1] : 1.0
    wStd[j] = wRaw[j] * s
    intercept += wRaw[j] * mu[j - 1]
  }
  wStd[0] = intercept
  return wStd
}

/**
 * Coordinate descent ElasticNet solver with soft-thresholding.
 */
export function solveElasticNet(
  X: Float64Array[],
  Y: number[] | Float64Array,
  l1Ratio = 0.3,
  alpha = 1e-3,
  maxIter = 200,
  tol = 1e-5
): Float64Array {
  const n = X.length
  if (n === 0) return new Float64Array(0)
  const p = X[0].length
  const w = new Float64Array(p)

  const l1 = alpha * l1Ratio * n
  const l2 = alpha * (1.0 - l1Ratio) * n

  // Precompute column squared norms
  const xNorms = new Float64Array(p)
  for (let j = 0; j < p; j++) {
    let sumSq = 0
    for (let i = 0; i < n; i++) sumSq += X[i][j] * X[i][j]
    xNorms[j] = sumSq > 1e-9 ? sumSq : 1e-9
  }

  // Precompute residuals
  const r = new Float64Array(n)
  for (let i = 0; i < n; i++) r[i] = Y[i]

  for (let iter = 0; iter < maxIter; iter++) {
    let maxChange = 0

    for (let j = 0; j < p; j++) {
      const wOld = w[j]

      // Compute rho_j = X_j^T (r + X_j * w_j)
      let rho = 0
      for (let i = 0; i < n; i++) {
        rho += X[i][j] * (r[i] + X[i][j] * wOld)
      }

      let wNew = 0
      if (j === 0) {
        // Intercept is unpenalized
        wNew = rho / xNorms[j]
      } else {
        // Soft-thresholding operator
        if (rho > l1) {
          wNew = (rho - l1) / (xNorms[j] + l2)
        } else if (rho < -l1) {
          wNew = (rho + l1) / (xNorms[j] + l2)
        } else {
          wNew = 0
        }
      }

      const diff = wNew - wOld
      if (Math.abs(diff) > maxChange) maxChange = Math.abs(diff)

      w[j] = wNew

      // Update residual r_i = r_i - X_ij * diff
      if (Math.abs(diff) > 1e-12) {
        for (let i = 0; i < n; i++) {
          r[i] -= X[i][j] * diff
        }
      }
    }

    if (maxChange < tol) break
  }

  return w
}

/**
 * Standard Ridge Solver via Normal Equations with unpenalized intercept and optional sample weighting.
 */
export function solveRidge(
  X: Float64Array[],
  Y: number[] | Float64Array,
  lambda = 1e-3,
  weights?: Float64Array | number[]
): Float64Array {
  const n = X.length
  if (n === 0) return new Float64Array(0)
  const p = X[0].length

  // Gram matrix XtX
  const XtX: number[][] = Array.from({ length: p }, () => new Array(p).fill(0))
  const XtY: number[] = new Array(p).fill(0)

  let totalWeight = 0
  for (let i = 0; i < n; i++) {
    const row = X[i]
    const y = Y[i]
    const w = weights ? weights[i] : 1.0
    totalWeight += w

    for (let j = 0; j < p; j++) {
      XtY[j] += row[j] * y * w
      for (let k = j; k < p; k++) {
        XtX[j][k] += row[j] * row[k] * w
      }
    }
  }

  // Symmetrize and add L2 ridge penalty (leaving intercept unpenalized)
  const effN = weights ? totalWeight : n
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < j; k++) {
      XtX[j][k] = XtX[k][j]
    }
    if (j > 0) {
      XtX[j][j] += lambda * effN
    }
  }

  // Gaussian elimination with partial pivoting
  const A = XtX.map((r) => [...r])
  const b = [...XtY]

  for (let col = 0; col < p; col++) {
    let maxRow = col
    let maxVal = Math.abs(A[col][col])
    for (let row = col + 1; row < p; row++) {
      if (Math.abs(A[row][col]) > maxVal) {
        maxVal = Math.abs(A[row][col])
        maxRow = row
      }
    }

    if (maxVal < 1e-12) continue

    if (maxRow !== col) {
      const tempR = A[col]
      A[col] = A[maxRow]
      A[maxRow] = tempR
      const tempB = b[col]
      b[col] = b[maxRow]
      b[maxRow] = tempB
    }

    const pivot = A[col][col]
    for (let row = col + 1; row < p; row++) {
      const factor = A[row][col] / pivot
      A[row][col] = 0
      for (let c = col + 1; c < p; c++) {
        A[row][c] -= factor * A[col][c]
      }
      b[row] -= factor * b[col]
    }
  }

  const w = new Float64Array(p)
  for (let row = p - 1; row >= 0; row--) {
    let sum = b[row]
    for (let col = row + 1; col < p; col++) {
      sum -= A[row][col] * w[col]
    }
    w[row] = Math.abs(A[row][row]) > 1e-12 ? sum / A[row][row] : 0
  }

  return w
}

/**
 * Multivariate anomaly filtering: detects and removes observations exhibiting anomalous
 * joint feature configurations (e.g. tracking glitches, sudden blink transitions, severe occlusion).
 * Discards the top `contamination` fraction (default 5%) of highest standardized distance.
 */
export function filterMultivariateOutliers(
  X: Float64Array[],
  Y: number[] | Float64Array,
  contamination = 0.05
): {
  cleanX: Float64Array[]
  cleanY: Float64Array
  inlierIndices: number[]
} {
  const n = X.length
  if (n <= 10 || contamination <= 0) {
    return {
      cleanX: X,
      cleanY: new Float64Array(Y),
      inlierIndices: Array.from({ length: n }, (_, i) => i),
    }
  }

  const p = X[0].length
  // Compute column means and standard deviations (ignoring intercept j=0)
  const mu = new Float64Array(p - 1)
  const sd = new Float64Array(p - 1)

  for (let j = 1; j < p; j++) {
    let sum = 0
    for (let i = 0; i < n; i++) sum += X[i][j]
    mu[j - 1] = sum / n

    let sumSq = 0
    for (let i = 0; i < n; i++) {
      const diff = X[i][j] - mu[j - 1]
      sumSq += diff * diff
    }
    const s = Math.sqrt(sumSq / n)
    sd[j - 1] = s < 1e-9 ? 1.0 : s
  }

  // Calculate squared standardized Euclidean distance to centroid
  const distances: { index: number; distSq: number }[] = []
  for (let i = 0; i < n; i++) {
    let dSq = 0
    for (let j = 1; j < p; j++) {
      const z = (X[i][j] - mu[j - 1]) / sd[j - 1]
      dSq += z * z
    }
    distances.push({ index: i, distSq: dSq })
  }

  // Sort distances ascending
  distances.sort((a, b) => a.distSq - b.distSq)
  const retainCount = Math.max(8, Math.floor(n * (1.0 - contamination)))
  const inlierSet = new Set(distances.slice(0, retainCount).map((d) => d.index))

  const cleanX: Float64Array[] = []
  const cleanYVals: number[] = []
  const inlierIndices: number[] = []

  for (let i = 0; i < n; i++) {
    if (inlierSet.has(i)) {
      cleanX.push(X[i])
      cleanYVals.push(Y[i])
      inlierIndices.push(i)
    }
  }

  return {
    cleanX,
    cleanY: new Float64Array(cleanYVals),
    inlierIndices,
  }
}
