/** Header status line showing API / LLM availability. */

import { getHealth } from "./api.js";
import { byId } from "./dom.js";

export async function refreshHealth() {
  const status = byId("health-status");
  try {
    const health = await getHealth();
    const llm = health.ollama ? "online" : "OFFLINE";
    status.textContent = `LLM ${health.llm_model} ${llm} | ${health.indexed_chunks} chunks indexed`;
    status.className = health.ollama ? "status ok" : "status bad";
    return health;
  } catch (error) {
    status.textContent = error.message;
    status.className = "status bad";
    return null;
  }
}
