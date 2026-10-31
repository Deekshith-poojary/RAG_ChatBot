/** Thin client for the RAG API (same origin, so no CORS or base-URL configuration is needed). */

const BASE_URL = "/api/v1";
const JSON_HEADERS = { "Content-Type": "application/json" };

export class ApiError extends Error {
  constructor(message) {
    super(message);
    this.name = "ApiError";
  }
}

async function errorDetail(response) {
  try {
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

async function send(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, options);
  } catch (error) {
    if (error.name === "AbortError") {
      throw error;
    }
    throw new ApiError("Cannot reach the API");
  }
  if (!response.ok) {
    throw new ApiError(await errorDetail(response));
  }
  return response;
}

async function requestJson(path, options) {
  const response = await send(path, options);
  return response.status === 204 ? null : response.json();
}

const postJson = (path, payload, signal) =>
  requestJson(path, { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(payload), signal });

export const getHealth = () => requestJson("/health");

export const listModels = () => requestJson("/llm/models");

export const listDocuments = () => requestJson("/documents");

export function uploadDocuments(files) {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file, file.name);
  }
  return requestJson("/documents", { method: "POST", body: form });
}

export const deleteDocument = (docId) =>
  requestJson(`/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });

export const chat = (payload, signal) => postJson("/chat", payload, signal);

/** Yield NDJSON events (`sources`, `token`, `done`, `error`) from /chat/stream as they arrive. */
export async function* chatStream(payload, signal) {
  const response = await send("/chat/stream", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
    signal,
  });
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += value;
    let newline = buffer.indexOf("\n");
    while (newline >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) {
        yield JSON.parse(line);
      }
      newline = buffer.indexOf("\n");
    }
  }
  if (buffer.trim()) {
    yield JSON.parse(buffer);
  }
}
