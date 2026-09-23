/**
 * fixation.ts — Anisotropic Uncertainty-Scaled I-DT Fixation Detector.
 *
 * Implements:
 * 1. Textbook sliding-window Dispersion-Threshold (I-DT) algorithm.
 * 2. Anisotropic dispersion thresholds scaled to tracker uncertainty (sigmaX, sigmaY).
 * 3. Seed-forward saccade transition: The sample that breaks a fixation is NOT
 *    discarded or averaged into the ending centroid; it seeds the next candidate fixation.
 * 4. Temporal gap and blink awareness (>120ms breaks fixation cleanly).
 */

export interface GazeSample {
  x: number
  y: number
  timestamp: number
  sigmaX?: number
  sigmaY?: number
}

export interface Fixation {
  id: string
  centroidX: number
  centroidY: number
  startTime: number
  endTime: number
  duration: number
  sampleCount: number
  dispersionX: number
  dispersionY: number
}

export interface FixationDetectorConfig {
  minDurationMs: number
  maxDurationMs: number
  baseDispersionXPx: number
  baseDispersionYPx: number
  dispersionSigmaMultiple: number
  maxGapMs: number
}

export const DEFAULT_FIXATION_CONFIG: FixationDetectorConfig = {
  minDurationMs: 140,
  maxDurationMs: 1000,
  baseDispersionXPx: 35,
  baseDispersionYPx: 25,
  dispersionSigmaMultiple: 1.5,
  maxGapMs: 120,
}

export class FixationDetector {
  private config: FixationDetectorConfig
  private window: GazeSample[] = []
  private activeFixation: Fixation | null = null
  private fixationCounter = 0

  constructor(config: Partial<FixationDetectorConfig> = {}) {
    this.config = { ...DEFAULT_FIXATION_CONFIG, ...config }
  }

  reset(): void {
    this.window = []
    this.activeFixation = null
  }

  processSample(sample: GazeSample): Fixation | null {
    // Gap check: blink or lost frame hard-breaks existing window
    if (this.window.length > 0) {
      const dt = sample.timestamp - this.window[this.window.length - 1].timestamp
      if (dt > this.config.maxGapMs) {
        const closed = this.activeFixation
        this.window = [sample]
        this.activeFixation = null
        return closed
      }
    }

    this.window.push(sample)

    // Check dispersion over current window
    const duration = this.window[this.window.length - 1].timestamp - this.window[0].timestamp
    const { dispX, dispY } = this.computeDispersion()

    const maxDispX = this.config.baseDispersionXPx + (sample.sigmaX || 0) * this.config.dispersionSigmaMultiple
    const maxDispY = this.config.baseDispersionYPx + (sample.sigmaY || 0) * this.config.dispersionSigmaMultiple

    if (dispX <= maxDispX && dispY <= maxDispY) {
      // Points cluster within threshold
      if (duration >= this.config.minDurationMs) {
        // Form or extend fixation
        const centroid = this.computeCentroid()
        this.activeFixation = {
          id: this.activeFixation?.id || `fix_${++this.fixationCounter}`,
          centroidX: centroid.x,
          centroidY: centroid.y,
          startTime: this.window[0].timestamp,
          endTime: sample.timestamp,
          duration,
          sampleCount: this.window.length,
          dispersionX: dispX,
          dispersionY: dispY,
        }

        // Split if exceeding max duration
        if (duration >= this.config.maxDurationMs) {
          const finished = this.activeFixation
          // Retain last half to seed next
          this.window = this.window.slice(Math.floor(this.window.length / 2))
          this.activeFixation = null
          return finished
        }
      }
      return null
    } else {
      // Window exceeded dispersion (saccade occurred)
      if (this.activeFixation) {
        const finished = this.activeFixation
        this.activeFixation = null
        // Seed-forward: pop the breaking sample to start next candidate window
        const breakingSample = this.window[this.window.length - 1]
        this.window = [breakingSample]
        return finished
      }

      // Slide window forward until dispersion recovers
      while (this.window.length > 1) {
        this.window.shift()
        const currentDisp = this.computeDispersion()
        if (currentDisp.dispX <= maxDispX && currentDisp.dispY <= maxDispY) {
          break
        }
      }
      return null
    }
  }

  private computeDispersion(): {
    dispX: number
    dispY: number
    minX: number
    maxX: number
    minY: number
    maxY: number
  } {
    let minX = Infinity, maxX = -Infinity
    let minY = Infinity, maxY = -Infinity

    for (const s of this.window) {
      if (s.x < minX) minX = s.x
      if (s.x > maxX) maxX = s.x
      if (s.y < minY) minY = s.y
      if (s.y > maxY) maxY = s.y
    }

    return {
      dispX: maxX - minX,
      dispY: maxY - minY,
      minX, maxX, minY, maxY,
    }
  }

  private computeCentroid(): { x: number; y: number } {
    let sumX = 0
    let sumY = 0
    for (const s of this.window) {
      sumX += s.x
      sumY += s.y
    }
    return {
      x: sumX / this.window.length,
      y: sumY / this.window.length,
    }
  }
}
