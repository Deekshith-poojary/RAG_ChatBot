/** Pure formatting helpers. */

const STATUS_LABELS = {
  queued: "Queued",
  extracting: "Extracting / OCR",
  embedding: "Embedding",
  ready: "Ready",
  failed: "Failed",
};

export const ACTIVE_STATUSES = new Set(["queued", "extracting", "embedding"]);

export const statusLabel = (status) => STATUS_LABELS[status] ?? status;

export const formatPages = (start, end) => (start === end ? `p. ${start}` : `pp. ${start}-${end}`);

export function formatDocumentMeta(doc) {
  const language = doc.language ? ` | ${doc.language}` : "";
  return `${doc.pages} pages (${doc.ocr_pages} OCR) | ${doc.chunks} chunks${language}`;
}
