/**
 * Converts a TipTap/ProseMirror JSON document (as produced by
 * editor.getJSON(), using the same node set StarterKit registers) into a
 * real .docx Blob via the `docx` package -- so "Download Local" produces a
 * Word document with actual heading styles and bold/italic runs instead of
 * raw Markdown syntax (`# `, `**`) opened as plain text.
 */
import type { JSONContent } from '@tiptap/core'
import {
  AlignmentType,
  Document,
  HeadingLevel,
  LevelFormat,
  Packer,
  Paragraph,
  TextRun,
} from 'docx'

const HEADING_LEVELS: Record<number, (typeof HeadingLevel)[keyof typeof HeadingLevel]> = {
  1: HeadingLevel.HEADING_1,
  2: HeadingLevel.HEADING_2,
  3: HeadingLevel.HEADING_3,
  4: HeadingLevel.HEADING_4,
  5: HeadingLevel.HEADING_5,
  6: HeadingLevel.HEADING_6,
}

const BULLET_LIST_REF = 'canvas-bullet-list'
const ORDERED_LIST_REF = 'canvas-ordered-list'
const MAX_LIST_LEVEL = 3

function inlineToRuns(nodes: JSONContent[] = []): TextRun[] {
  const runs: TextRun[] = []
  for (const node of nodes) {
    if (node.type === 'hardBreak') {
      runs.push(new TextRun({ text: '', break: 1 }))
      continue
    }
    if (node.type !== 'text' || !node.text) continue
    const marks = new Set((node.marks ?? []).map((mark) => mark.type))
    runs.push(
      new TextRun({
        text: node.text,
        bold: marks.has('bold'),
        italics: marks.has('italic'),
        strike: marks.has('strike'),
        font: marks.has('code') ? 'Consolas' : undefined,
      })
    )
  }
  // docx requires at least one run per paragraph to render an empty line.
  return runs.length > 0 ? runs : [new TextRun({ text: '' })]
}

function listToParagraphs(list: JSONContent, level = 0): Paragraph[] {
  const ordered = list.type === 'orderedList'
  const reference = ordered ? ORDERED_LIST_REF : BULLET_LIST_REF
  const boundedLevel = Math.min(level, MAX_LIST_LEVEL)
  const paragraphs: Paragraph[] = []

  for (const item of list.content ?? []) {
    for (const child of item.content ?? []) {
      if (child.type === 'paragraph') {
        paragraphs.push(
          new Paragraph({
            children: inlineToRuns(child.content),
            numbering: { reference, level: boundedLevel },
          })
        )
      } else if (child.type === 'bulletList' || child.type === 'orderedList') {
        paragraphs.push(...listToParagraphs(child, level + 1))
      }
    }
  }
  return paragraphs
}

function blockquoteToParagraphs(node: JSONContent): Paragraph[] {
  const paragraphs: Paragraph[] = []
  for (const child of node.content ?? []) {
    if (child.type !== 'paragraph') continue
    paragraphs.push(
      new Paragraph({
        children: inlineToRuns(child.content),
        indent: { left: 480 },
        border: { left: { color: '999999', space: 8, style: 'single', size: 6 } },
      })
    )
  }
  return paragraphs
}

function codeBlockToParagraphs(node: JSONContent): Paragraph[] {
  const text = (node.content ?? []).map((child) => child.text ?? '').join('')
  return text.split('\n').map(
    (line) =>
      new Paragraph({
        children: [new TextRun({ text: line, font: 'Consolas' })],
      })
  )
}

function docNodeToBlocks(node: JSONContent): Paragraph[] {
  switch (node.type) {
    case 'heading': {
      const level = (node.attrs?.level as number) ?? 1
      return [
        new Paragraph({
          heading: HEADING_LEVELS[level] ?? HeadingLevel.HEADING_1,
          children: inlineToRuns(node.content),
        }),
      ]
    }
    case 'paragraph':
      return [new Paragraph({ children: inlineToRuns(node.content) })]
    case 'bulletList':
    case 'orderedList':
      return listToParagraphs(node)
    case 'blockquote':
      return blockquoteToParagraphs(node)
    case 'codeBlock':
      return codeBlockToParagraphs(node)
    case 'horizontalRule':
      return [
        new Paragraph({
          children: [],
          border: { bottom: { color: 'CCCCCC', space: 1, style: 'single', size: 6 } },
        }),
      ]
    default:
      return []
  }
}

export function tiptapJsonToDocxBlocks(doc: JSONContent): Paragraph[] {
  return (doc.content ?? []).flatMap(docNodeToBlocks)
}

export async function tiptapJsonToDocxBlob(doc: JSONContent, title: string): Promise<Blob> {
  const document = new Document({
    title,
    numbering: {
      config: [
        {
          reference: BULLET_LIST_REF,
          levels: Array.from({ length: MAX_LIST_LEVEL + 1 }, (_, level) => ({
            level,
            format: LevelFormat.BULLET,
            text: '•',
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720 * (level + 1), hanging: 360 } } },
          })),
        },
        {
          reference: ORDERED_LIST_REF,
          levels: Array.from({ length: MAX_LIST_LEVEL + 1 }, (_, level) => ({
            level,
            format: LevelFormat.DECIMAL,
            text: `%${level + 1}.`,
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720 * (level + 1), hanging: 360 } } },
          })),
        },
      ],
    },
    sections: [
      {
        properties: {},
        children: tiptapJsonToDocxBlocks(doc),
      },
    ],
  })

  return Packer.toBlob(document)
}
