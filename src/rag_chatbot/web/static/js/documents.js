/** Document upload, status list (polled while ingestion runs) and the document filter. */

import { deleteDocument, listDocuments, uploadDocuments } from "./api.js";
import { byId, h } from "./dom.js";
import { ACTIVE_STATUSES, formatDocumentMeta, statusLabel } from "./format.js";
import { state } from "./state.js";

const POLL_INTERVAL_MS = 3000;

let els;
let pollTimer = null;
let renderedSignature = "";

const isActive = (doc) => ACTIVE_STATUSES.has(doc.status);

export function initDocuments() {
  els = {
    dropzone: byId("dropzone"),
    fileInput: byId("file-input"),
    messages: byId("upload-messages"),
    list: byId("doc-list"),
    count: byId("doc-count"),
    empty: byId("doc-empty"),
    filter: byId("doc-filter"),
  };

  els.fileInput.addEventListener("change", () => {
    const files = [...els.fileInput.files];
    els.fileInput.value = "";
    handleFiles(files);
  });

  for (const type of ["dragenter", "dragover"]) {
    els.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      els.dropzone.classList.add("dragging");
    });
  }
  for (const type of ["dragleave", "drop"]) {
    els.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      els.dropzone.classList.remove("dragging");
    });
  }
  els.dropzone.addEventListener("drop", (event) => handleFiles([...event.dataTransfer.files]));

  return refreshDocuments();
}

export async function refreshDocuments() {
  clearTimeout(pollTimer);
  let failed = false;
  try {
    state.documents = await listDocuments();
    render();
  } catch {
    failed = true;
  }
  if (failed || state.documents.some(isActive)) {
    pollTimer = setTimeout(refreshDocuments, POLL_INTERVAL_MS);
  }
}

function showMessages(messages) {
  els.messages.replaceChildren(...messages.map(({ kind, text }) => h("li", { class: kind }, text)));
}

async function handleFiles(files) {
  const pdfs = files.filter((file) => file.name.toLowerCase().endsWith(".pdf"));
  const messages = [];
  if (pdfs.length < files.length) {
    messages.push({ kind: "error", text: "Only PDF files are accepted." });
  }
  if (pdfs.length === 0) {
    showMessages(messages);
    return;
  }
  showMessages([...messages, { kind: "info", text: `Uploading ${pdfs.length} file(s)...` }]);

  try {
    const results = await uploadDocuments(pdfs);
    for (const result of results) {
      if (result.status === "rejected") {
        messages.push({ kind: "error", text: `${result.filename}: ${result.error}` });
      } else if (result.status === "duplicate") {
        messages.push({ kind: "info", text: `${result.filename}: already ingested` });
      } else {
        messages.push({ kind: "success", text: `${result.filename}: queued for ingestion` });
      }
    }
  } catch (error) {
    messages.push({ kind: "error", text: error.message });
  }
  showMessages(messages);
  await refreshDocuments();
}

async function handleDelete(doc) {
  if (!window.confirm(`Delete "${doc.filename}" and its indexed content?`)) {
    return;
  }
  try {
    await deleteDocument(doc.doc_id);
  } catch (error) {
    showMessages([{ kind: "error", text: error.message }]);
  }
  await refreshDocuments();
}

function stateClass(status) {
  if (status === "ready" || status === "failed") {
    return status;
  }
  return "working";
}

function renderDocument(doc) {
  const active = isActive(doc);
  return h(
    "li",
    { class: "doc-item" },
    h("div", { class: "doc-name" }, doc.filename),
    h(
      "div",
      { class: "doc-meta" },
      h("span", { class: `state ${stateClass(doc.status)}` }, statusLabel(doc.status)),
      ` | ${formatDocumentMeta(doc)}`,
    ),
    active ? h("progress", { max: 1, value: doc.progress, "aria-label": `Progress for ${doc.filename}` }) : null,
    doc.status === "failed" ? h("p", { class: "doc-error" }, doc.error ?? "Ingestion failed") : null,
    active
      ? null
      : h(
          "button",
          {
            type: "button",
            class: "danger",
            "aria-label": `Delete ${doc.filename}`,
            onClick: () => handleDelete(doc),
          },
          "Delete",
        ),
  );
}

function renderFilter() {
  const ready = state.documents.filter((doc) => doc.status === "ready");
  const readyIds = new Set(ready.map((doc) => doc.doc_id));
  for (const id of state.selectedDocIds) {
    if (!readyIds.has(id)) {
      state.selectedDocIds.delete(id);
    }
  }
  els.filter.replaceChildren(
    ...ready.map((doc) =>
      h(
        "label",
        {},
        h("input", {
          type: "checkbox",
          checked: state.selectedDocIds.has(doc.doc_id),
          onChange: (event) => {
            if (event.target.checked) {
              state.selectedDocIds.add(doc.doc_id);
            } else {
              state.selectedDocIds.delete(doc.doc_id);
            }
          },
        }),
        doc.filename,
      ),
    ),
  );
}

function render() {
  const signature = JSON.stringify(state.documents);
  if (signature === renderedSignature) {
    return;
  }
  renderedSignature = signature;

  els.count.textContent = String(state.documents.length);
  els.empty.hidden = state.documents.length > 0;
  els.list.replaceChildren(...state.documents.map(renderDocument));
  renderFilter();
}
