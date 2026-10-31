/** Chat settings panel. */

import { listModels } from "./api.js";
import { byId } from "./dom.js";
import { state } from "./state.js";

let els;

export async function initSettings({ defaultModel, rerankAvailable }) {
  els = {
    form: byId("settings-form"),
    model: byId("model-select"),
    topK: byId("top-k"),
    topKValue: byId("top-k-value"),
    rerank: byId("rerank"),
    stream: byId("stream"),
  };

  els.form.addEventListener("submit", (event) => event.preventDefault());
  els.topK.addEventListener("input", () => {
    els.topKValue.textContent = els.topK.value;
  });
  if (!rerankAvailable) {
    els.rerank.checked = false;
    els.rerank.disabled = true;
  }
  await populateModels(defaultModel);
}

async function populateModels(defaultModel) {
  let names = [];
  try {
    names = (await listModels()).map((model) => model.name);
  } catch {
    // The endpoint may be offline; fall back to the configured model only.
  }
  if (defaultModel && !names.includes(defaultModel)) {
    names.unshift(defaultModel);
  }
  els.model.replaceChildren(
    ...names.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      return option;
    }),
  );
  if (defaultModel) {
    els.model.value = defaultModel;
  }
}

export function getChatOptions() {
  return {
    model: els.model.value || null,
    top_k: Number(els.topK.value),
    rerank: els.rerank.checked,
    stream: els.stream.checked,
    doc_ids: state.selectedDocIds.size > 0 ? [...state.selectedDocIds] : null,
  };
}
