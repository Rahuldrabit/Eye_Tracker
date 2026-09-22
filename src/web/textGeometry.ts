/**
 * textGeometry.ts — Zero-DOM Word Boundary Indexing via Intl.Segmenter & Range.getClientRects().
 *
 * Extracts pixel bounding rects for every word in a text node directly from the browser layout engine.
 * Completely avoids wrapping words in <span> tags, preventing DOM bloat and layout thrashing.
 */

import type { MeasuredWordBox } from './lineSnapper'

export function extractWordBoxesFromElement(container: HTMLElement): MeasuredWordBox[] {
  const wordBoxes: MeasuredWordBox[] = []
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null)
  const segmenter = new Intl.Segmenter('en', { granularity: 'word' })

  let textNode: Text | null = walker.nextNode() as Text | null
  let globalWordIndex = 0

  while (textNode) {
    const text = textNode.nodeValue || ''
    const segments = Array.from(segmenter.segment(text))

    for (const seg of segments) {
      if (!seg.isWordLike) continue

      const range = document.createRange()
      try {
        range.setStart(textNode, seg.index)
        range.setEnd(textNode, seg.index + seg.segment.length)

        const clientRects = Array.from(range.getClientRects())
        if (clientRects.length > 0) {
          wordBoxes.push({
            wordIndex: globalWordIndex++,
            text: seg.segment,
            rects: clientRects,
          })
        }
      } catch {
        // Skip detached ranges
      }
    }

    textNode = walker.nextNode() as Text | null
  }

  return wordBoxes
}
