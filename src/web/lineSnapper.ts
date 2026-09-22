/**
 * lineSnapper.ts — Soft Gaussian Line-Attractor for Reading Gaze Stabilization.
 *
 * Prevents vertical gaze jitter from randomly hopping between adjacent text lines
 * while preserving 100% fluid, zero-lag horizontal reading saccades.
 */

export interface MeasuredWordBox {
  wordIndex: number
  text: string
  rects: DOMRect[]
}

export interface TextLineBand {
  lineIndex: number
  top: number
  bottom: number
  centerY: number
  height: number
  left: number
  right: number
  wordBoxes: MeasuredWordBox[]
}

export class ReadingLineSnapper {
  private lines: TextLineBand[] = []
  private sigma = 14.0 // Gaussian attractor width in pixels (~half line height)

  updateFromWordBoxes(boxes: MeasuredWordBox[]): void {
    if (boxes.length === 0) {
      this.lines = []
      return
    }

    interface RectItem {
      box: MeasuredWordBox
      rect: DOMRect
    }

    const items: RectItem[] = []
    for (const b of boxes) {
      for (const r of b.rects) {
        if (r.width > 0 && r.height > 0) {
          items.push({ box: b, rect: r })
        }
      }
    }

    if (items.length === 0) {
      this.lines = []
      return
    }

    // Sort vertically
    items.sort((a, b) => a.rect.top - b.rect.top)

    // Group rects whose midpoints are within ~14px
    const lineGroups: RectItem[][] = []
    for (const item of items) {
      const midY = (item.rect.top + item.rect.bottom) / 2.0
      let placed = false

      for (const group of lineGroups) {
        const groupMidY =
          group.reduce((acc, it) => acc + (it.rect.top + it.rect.bottom) / 2.0, 0) / group.length
        if (Math.abs(midY - groupMidY) < 14) {
          group.push(item)
          placed = true
          break
        }
      }

      if (!placed) {
        lineGroups.push([item])
      }
    }

    lineGroups.sort((a, b) => {
      const midA = a.reduce((acc, it) => acc + it.rect.top, 0) / a.length
      const midB = b.reduce((acc, it) => acc + it.rect.top, 0) / b.length
      return midA - midB
    })

    this.lines = lineGroups.map((grp, idx) => {
      let top = Infinity, bottom = -Infinity
      let left = Infinity, right = -Infinity
      const wordBoxSet = new Set<MeasuredWordBox>()

      for (const it of grp) {
        if (it.rect.top < top) top = it.rect.top
        if (it.rect.bottom > bottom) bottom = it.rect.bottom
        if (it.rect.left < left) left = it.rect.left
        if (it.rect.right > right) right = it.rect.right
        wordBoxSet.add(it.box)
      }

      return {
        lineIndex: idx,
        top,
        bottom,
        centerY: (top + bottom) / 2.0,
        height: bottom - top,
        left,
        right,
        wordBoxes: Array.from(wordBoxSet),
      }
    })
  }

  getLineCount(): number {
    return this.lines.length
  }

  snapGaze(x: number, y: number): { x: number; y: number; isSnapped: boolean; lineIndex?: number } {
    if (this.lines.length === 0) {
      return { x, y, isSnapped: false }
    }

    let bestLine: TextLineBand | null = null
    let bestDist = Infinity

    for (const line of this.lines) {
      // Horizontal margin gate
      if (x >= line.left - 50 && x <= line.right + 50) {
        const dist = Math.abs(y - line.centerY)
        if (dist < bestDist) {
          bestDist = dist
          bestLine = line
        }
      }
    }

    if (bestLine && bestDist < 30.0) {
      // Gaussian soft pull: w = exp(-0.5 * (d / sigma)^2)
      const weight = Math.exp(-0.5 * Math.pow(bestDist / this.sigma, 2))
      const snappedY = y * (1.0 - weight) + bestLine.centerY * weight
      return {
        x,
        y: snappedY,
        isSnapped: true,
        lineIndex: bestLine.lineIndex,
      }
    }

    return { x, y, isSnapped: false }
  }
}
