/**
 * bayesianAOI.ts — Two-Stage Hierarchical Bayesian Word Attribution with Preferred Viewing Location (PVL).
 *
 * Stage 1: Line Selection Posterior
 * L_i = exp(-0.5 * ((fy - y_i) / sigma_y)^2) * exp(-0.5 * (horizontal_gate / sigma_x)^2)
 *
 * Stage 2: Word Attribution Posterior with Human Preferred Viewing Location (PVL = 0.4)
 * target(word) = left(word) + 0.4 * width(word)  <-- Human foveal landing bias
 * P(word) = exp(-0.5 * ((fx - target(word)) / sigma_x)^2) / sum(scores)
 */

import type { TextLineBand, MeasuredWordBox } from './lineSnapper'

export interface AttributionResult {
  level: 'word' | 'wordSpan' | 'line' | 'lineBand' | 'none'
  wordIndex?: number
  wordText?: string
  confidence: number
  lineIndex?: number
}

export function attributeFixationToText(
  fx: number,
  fy: number,
  sigmaX: number,
  sigmaY: number,
  lines: TextLineBand[]
): AttributionResult {
  if (lines.length === 0) return { level: 'none', confidence: 0 }

  const sX = Math.max(10, sigmaX)
  const sY = Math.max(8, sigmaY)

  // Stage 1: Line posterior
  const lineScores: { line: TextLineBand; score: number }[] = []
  let sumLineScore = 0

  for (const line of lines) {
    const dy = (fy - line.centerY) / sY
    const vertScore = Math.exp(-0.5 * dy * dy)

    // Horizontal margin gate
    const hDist = Math.max(0, line.left - fx, fx - line.right)
    const horizScore = Math.exp(-0.5 * Math.pow(hDist / sX, 2))

    const totalScore = vertScore * horizScore
    lineScores.push({ line, score: totalScore })
    sumLineScore += totalScore
  }

  if (sumLineScore < 1e-6) return { level: 'none', confidence: 0 }

  lineScores.sort((a, b) => b.score - a.score)
  const bestLine = lineScores[0]
  const pBestLine = bestLine.score / sumLineScore

  // Runner-up ratio
  const runnerUpScore = lineScores.length > 1 ? lineScores[1].score : 0
  const runnerUpRatio = bestLine.score / Math.max(1e-6, runnerUpScore)

  if (pBestLine < 0.5 || runnerUpRatio < 1.4) {
    return {
      level: 'lineBand',
      confidence: pBestLine,
      lineIndex: bestLine.line.lineIndex,
    }
  }

  // Stage 2: Word within best line
  const words = bestLine.line.wordBoxes
  if (words.length === 0) {
    return {
      level: 'line',
      confidence: pBestLine,
      lineIndex: bestLine.line.lineIndex,
    }
  }

  const wordScores: { box: MeasuredWordBox; score: number }[] = []
  let sumWordScore = 0

  for (const box of words) {
    const rect = box.rects[0]
    if (!rect) continue

    // Preferred Viewing Location (PVL) = 0.4 of word width (left of center)
    const pvlTargetX = rect.left + 0.4 * rect.width
    const dx = (fx - pvlTargetX) / sX
    const score = Math.exp(-0.5 * dx * dx)

    wordScores.push({ box, score })
    sumWordScore += score
  }

  if (sumWordScore < 1e-6) {
    return {
      level: 'line',
      confidence: pBestLine,
      lineIndex: bestLine.line.lineIndex,
    }
  }

  wordScores.sort((a, b) => b.score - a.score)
  const bestWord = wordScores[0]
  const pBestWord = bestWord.score / sumWordScore

  if (pBestWord >= 0.48) {
    return {
      level: 'word',
      wordIndex: bestWord.box.wordIndex,
      wordText: bestWord.box.text,
      confidence: pBestWord,
      lineIndex: bestLine.line.lineIndex,
    }
  }

  return {
    level: 'wordSpan',
    wordIndex: bestWord.box.wordIndex,
    wordText: bestWord.box.text,
    confidence: pBestWord,
    lineIndex: bestLine.line.lineIndex,
  }
}
