import { initChat } from "./chat.js";
import { initDocuments } from "./documents.js";
import { refreshHealth } from "./health.js";
import { getChatOptions, initSettings } from "./settings.js";

const HEALTH_INTERVAL_MS = 15000;

async function start() {
  const documentsReady = initDocuments();
  const health = await refreshHealth();
  await initSettings({
    defaultModel: health?.llm_model ?? null,
    rerankAvailable: health?.reranker ?? true,
  });
  initChat(getChatOptions);
  setInterval(refreshHealth, HEALTH_INTERVAL_MS);
  await documentsReady;
}

start();
