/** Minimal DOM helpers. All text goes through text nodes, so model output can never inject markup. */

export const byId = (id) => {
  const node = document.getElementById(id);
  if (!node) {
    throw new Error(`Missing element #${id}`);
  }
  return node;
};

/**
 * Create an element. Props named on* become event listeners, "class" sets className,
 * known DOM properties are assigned, everything else becomes an attribute.
 */
export function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) {
      continue;
    }
    if (key === "class") {
      node.className = value;
    } else if (key === "dataset") {
      Object.assign(node.dataset, value);
    } else if (key.startsWith("on")) {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key in node) {
      node[key] = value;
    } else {
      node.setAttribute(key, value);
    }
  }
  node.append(...children.filter((child) => child !== null && child !== undefined && child !== false));
  return node;
}
