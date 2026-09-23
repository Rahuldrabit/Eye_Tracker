/**
 * oneEuroFilter.ts — Speed-Adaptive Low-Pass Filter (Casiez et al., CHI 2012).
 *
 * Simultaneously eliminates low-speed fixation jitter (low cutoff) and
 * tracks high-speed ballistic saccades with zero perceptible lag (adaptive high cutoff).
 */

export class OneEuroFilter2D {
  private minCutoff: number
  private beta: number
  private dCutoff: number
  private xHatPrev: number | null = null
  private yHatPrev: number | null = null
  private dxHatPrev = 0
  private dyHatPrev = 0
  private tPrev: number | null = null

  constructor(minCutoff = 1.0, beta = 0.05, dCutoff = 1.0) {
    this.minCutoff = minCutoff
    this.beta = beta
    this.dCutoff = dCutoff
  }

  reset(): void {
    this.xHatPrev = null
    this.yHatPrev = null
    this.dxHatPrev = 0
    this.dyHatPrev = 0
    this.tPrev = null
  }

  filter(x: number, y: number, timestampMs: number = Date.now()): { x: number; y: number } {
    if (this.tPrev === null || this.xHatPrev === null || this.yHatPrev === null) {
      this.xHatPrev = x
      this.yHatPrev = y
      this.tPrev = timestampMs
      return { x, y }
    }

    const dt = Math.max(1e-4, (timestampMs - this.tPrev) / 1000.0)
    this.tPrev = timestampMs

    // 1. Raw discrete derivatives
    const alphaD = this.computeAlpha(this.dCutoff, dt)
    const dx = (x - this.xHatPrev) / dt
    const dy = (y - this.yHatPrev) / dt

    // 2. Filtered derivatives
    const dxHat = alphaD * dx + (1.0 - alphaD) * this.dxHatPrev
    const dyHat = alphaD * dy + (1.0 - alphaD) * this.dyHatPrev
    this.dxHatPrev = dxHat
    this.dyHatPrev = dyHat

    // 3. Dynamic cutoff frequency tracking velocity
    const cutoffX = this.minCutoff + this.beta * Math.abs(dxHat)
    const cutoffY = this.minCutoff + this.beta * Math.abs(dyHat)

    // 4. Adaptively filter positions
    const alphaX = this.computeAlpha(cutoffX, dt)
    const alphaY = this.computeAlpha(cutoffY, dt)

    const xHat = alphaX * x + (1.0 - alphaX) * this.xHatPrev
    const yHat = alphaY * y + (1.0 - alphaY) * this.yHatPrev

    this.xHatPrev = xHat
    this.yHatPrev = yHat

    return { x: xHat, y: yHat }
  }

  private computeAlpha(cutoff: number, dt: number): number {
    const tau = 1.0 / (2.0 * Math.PI * cutoff)
    return 1.0 / (1.0 + tau / dt)
  }
}
