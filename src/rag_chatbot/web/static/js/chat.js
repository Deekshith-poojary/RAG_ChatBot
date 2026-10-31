/** Conversation view: sends questions, streams answers and renders citations and sources. */

import { ApiError, chat, chatStream } from "./api.js";
import { byId, h } from "./dom.js";
import { formatPages } from "./format.js";

const CITATION_PATTERN = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
const PIN_THRESHOLD_PX = 80;
const FLASH_MS = 1500;

let els;
let getOptions;
let controller = null;
let messageCounter = 0;

export function initChat(optionsProvider) {
  getOptions = optionsProvider;
  els = {
    log: byId("chat-log"),
    empty: byId("chat-empty"),
    form: byId("chat-form"),
    question: byId("question"),
    send: byId("send-btn"),
    stop: byId("stop-btn"),
    clear: byId("clear-btn"),
  };

  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    submit();
  });
  els.question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      els.form.requestSubmit();
    }
  });
  els.stop.addEventListener("click", () => controller?.abort());
  els.clear.addEventListener("click", clearChat);
}

function setBusy(busy) {
  els.send.disabled = busy;
  els.stop.hidden = !busy;
}

function clearChat() {
  controller?.abort();
  els.log.replaceChildren(els.empty);
  els.empty.hidden = false;
}

/** Run a DOM mutation and keep the log scrolled to the bottom if the user was already there. */
function keepPinned(mutate) {
  const { log } = els;
  const pinned = log.scrollHeight - log.scrollTop - log.clientHeight < PIN_THRESHOLD_PX;
  mutate();
  if (pinned) {
    log.scrollTop = log.scrollHeight;
  }
}

function appendUserMessage(text) {
  els.empty.hidden = true;
  keepPinned(() => els.log.append(h("article", { class: "message user" }, h("div", { class: "body" }, text))));
}

function sourceElementId(messageId, index) {
  return `src-${messageId}-${index}`;
}

function citationChip(number, sourceCount, onSelect) {
  const valid = number >= 1 && number <= sourceCount;
  return h(
    "button",
    {
      type: "button",
      class: valid ? "cite" : "cite invalid",
      title: valid ? `Show source ${number}` : "No such source",
      onClick: valid ? () => onSelect(number) : null,
    },
    String(number),
  );
}

function renderAnswer(container, text, sourceCount, onSelect) {
  const nodes = [];
  let last = 0;
  for (const match of text.matchAll(CITATION_PATTERN)) {
    nodes.push(text.slice(last, match.index));
    for (const part of match[1].split(",")) {
      nodes.push(citationChip(Number(part.trim()), sourceCount, onSelect));
    }
    last = match.index + match[0].length;
  }
  nodes.push(text.slice(last));
  container.replaceChildren(...nodes);
}

function renderSources(messageId, sources) {
  const items = sources.map((source) =>
    h(
      "li",
      { id: sourceElementId(messageId, source.index) },
      h(
        "span",
        { class: "source-title" },
        `${source.filename} - ${formatPages(source.page_start, source.page_end)}`,
      ),
      ` (score ${source.score.toFixed(3)})`,
      h("p", { class: "snippet" }, `${source.snippet}...`),
    ),
  );
  return h("details", { class: "sources" }, h("summary", {}, `Sources (${sources.length})`), h("ol", {}, ...items));
}

function createAssistantMessage() {
  const id = ++messageCounter;
  const body = h("div", { class: "body" });
  const meta = h("div", { class: "meta" });
  const node = h("article", { class: "message assistant pending", "aria-busy": "true" }, body, meta);
  keepPinned(() => els.log.append(node));

  let text = "";
  let sources = [];
  let sourcesNode = null;

  const selectSource = (index) => {
    const item = document.getElementById(sourceElementId(id, index));
    if (!item) {
      return;
    }
    sourcesNode.open = true;
    item.scrollIntoView({ block: "nearest" });
    item.classList.add("flash");
    setTimeout(() => item.classList.remove("flash"), FLASH_MS);
  };

  const redraw = () => keepPinned(() => renderAnswer(body, text, sources.length, selectSource));

  return {
    setSources(list) {
      sources = list;
      if (sources.length === 0) {
        return;
      }
      sourcesNode = renderSources(id, sources);
      keepPinned(() => meta.append(sourcesNode));
      redraw();
    },
    appendText(piece) {
      text += piece;
      redraw();
    },
    finish({ citations }) {
      for (const index of citations.cited) {
        document.getElementById(sourceElementId(id, index))?.classList.add("cited");
      }
      if (citations.invalid.length > 0) {
        const invalid = citations.invalid.join(", ");
        keepPinned(() => meta.append(h("p", { class: "warning" }, `The answer cites sources that do not exist: ${invalid}`)));
      }
    },
    fail(message) {
      keepPinned(() => meta.append(h("p", { class: "error" }, message)));
    },
    close() {
      node.classList.remove("pending");
      node.removeAttribute("aria-busy");
    },
  };
}

async function streamAnswer(view, payload, signal) {
  for await (const event of chatStream(payload, signal)) {
    switch (event.type) {
      case "sources":
        view.setSources(event.sources);
        break;
      case "token":
        view.appendText(event.text);
        break;
      case "done":
        view.finish(event);
        break;
      case "error":
        throw new ApiError(event.message);
      default:
        break;
    }
  }
}

async function blockingAnswer(view, payload, signal) {
  const result = await chat(payload, signal);
  view.setSources(result.sources);
  view.appendText(result.answer);
  view.finish(result);
}

async function submit() {
  const question = els.question.value.trim();
  if (!question || controller) {
    return;
  }
  els.question.value = "";
  appendUserMessage(question);

  const view = createAssistantMessage();
  const { stream, ...filters } = getOptions();
  const payload = { question, ...filters };
  controller = new AbortController();
  setBusy(true);

  try {
    await (stream ? streamAnswer : blockingAnswer)(view, payload, controller.signal);
  } catch (error) {
    view.fail(error.name === "AbortError" ? "Stopped." : error.message);
  } finally {
    view.close();
    controller = null;
    setBusy(false);
    els.question.focus();
  }
}
