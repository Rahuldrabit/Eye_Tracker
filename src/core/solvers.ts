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
 * Standard Ridge Solver via Normal Equations with unpenalized intercept.
 */
export function solveRidge(
  X: Float64Array[],
  Y: number[] | Float64Array,
  lambda = 1e-3
): Float64Array {
  const n = X.length
  const p = X[0].length

  // Gram matrix XtX
  const XtX: number[][] = Array.from({ length: p }, () => new Array(p).fill(0))
  const XtY: number[] = new Array(p).fill(0)

  for (let i = 0; i < n; i++) {
    const row = X[i]
    const y = Y[i]
    for (let j = 0; j < p; j++) {
      XtY[j] += row[j] * y
      for (let k = j; k < p; k++) {
        XtX[j][k] += row[j] * row[k]
      }
    }
  }

  // Symmetrize and add L2 ridge penalty (leaving intercept unpenalized)
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < j; k++) {
      XtX[j][k] = XtX[k][j]
    }
    if (j > 0) {
      XtX[j][j] += lambda * n
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
