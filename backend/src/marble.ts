/**
 * World Labs Marble API client: single image -> explorable 3D Gaussian world.
 *
 *   base : https://api.worldlabs.ai/marble/v1
 *   auth : header `WLT-Api-Key: <key>`   (NOT a Bearer token)
 *   gen  : POST /worlds:generate { display_name, model, world_prompt }
 *   poll : GET  /worlds/{id} until a terminal state, then read SPZ output URLs
 *
 * The key is read from WORLDLABS_API_KEY and never logged. Exact response field
 * names aren't publicly documented, so we log full bodies (key-free) in CI and
 * extract the splat URL by pattern (any *.spz) to stay robust to schema drift.
 */
const BASE = "https://api.worldlabs.ai/marble/v1";

function authHeaders(): Record<string, string> {
  const key = process.env.WORLDLABS_API_KEY;
  if (!key) throw new Error("WORLDLABS_API_KEY not set");
  return { "WLT-Api-Key": key, "Content-Type": "application/json" };
}

/** Recursively find the first string value that looks like an SPZ/PLY splat URL. */
export function extractSplatUrl(obj: unknown, prefer = ".spz"): string | undefined {
  const hits: string[] = [];
  const walk = (v: unknown) => {
    if (typeof v === "string") {
      if (/^https?:\/\/\S+\.(spz|ply|splat|sog|ksplat)(\?|$)/i.test(v)) hits.push(v);
    } else if (Array.isArray(v)) {
      v.forEach(walk);
    } else if (v && typeof v === "object") {
      Object.values(v as Record<string, unknown>).forEach(walk);
    }
  };
  walk(obj);
  return hits.find((u) => u.toLowerCase().includes(prefer)) ?? hits[0];
}

/** Find an id we can poll on from the generate response. */
function extractWorldId(obj: any): string | undefined {
  return (
    obj?.world?.id ?? obj?.id ?? obj?.world_id ?? obj?.name?.split?.("/").pop() ?? obj?.operation?.id
  );
}

function extractState(obj: any): string {
  return String(obj?.state ?? obj?.status ?? obj?.world?.state ?? obj?.world?.status ?? "").toUpperCase();
}

export interface MarbleResult {
  splatUrl: string;
  worldId: string;
  raw: unknown;
}

/**
 * Generate a world from a public image URL and wait for it to finish.
 * @param model "marble-1.1" (full) or "marble-1.0-draft" (cheap, for testing)
 */
export async function generateWorldFromImage(
  imageUrl: string,
  opts: { model?: string; displayName?: string; deadline: number; pollMs?: number }
): Promise<MarbleResult> {
  const model = opts.model ?? "marble-1.1";
  const body = {
    display_name: opts.displayName ?? "diorama",
    model,
    world_prompt: { type: "image", image_prompt: { source: "uri", uri: imageUrl } },
  };

  console.error("Marble: POST /worlds:generate", JSON.stringify({ model, image_url: imageUrl }));
  const res = await fetch(`${BASE}/worlds:generate`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify(body),
  });
  const genText = await res.text();
  console.error("Marble generate HTTP", res.status, "body:", genText.slice(0, 1200));
  if (!res.ok) throw new Error(`Marble generate ${res.status}: ${genText.slice(0, 300)}`);

  let gen: any;
  try { gen = JSON.parse(genText); } catch { gen = {}; }

  // Some APIs return the finished world inline; check first.
  let splat = extractSplatUrl(gen);
  const worldId = extractWorldId(gen);
  if (splat) return { splatUrl: splat, worldId: worldId ?? "", raw: gen };
  if (!worldId) throw new Error("Marble: no world id in generate response");

  // Poll until a terminal state.
  const pollMs = opts.pollMs ?? 5000;
  while (Date.now() < opts.deadline) {
    await new Promise((r) => setTimeout(r, pollMs));
    const pr = await fetch(`${BASE}/worlds/${worldId}`, { headers: authHeaders() });
    const pText = await pr.text();
    if (!pr.ok) { console.error("Marble poll HTTP", pr.status, pText.slice(0, 300)); continue; }
    let world: any; try { world = JSON.parse(pText); } catch { continue; }
    const state = extractState(world);
    console.error("Marble poll state:", state || "(none)");
    splat = extractSplatUrl(world);
    if (splat) return { splatUrl: splat, worldId, raw: world };
    if (/(FAIL|ERROR|CANCEL)/.test(state)) {
      console.error("Marble world body:", pText.slice(0, 1200));
      throw new Error(`Marble world ${worldId} ${state}`);
    }
  }
  throw new Error("Marble: timed out waiting for world");
}
