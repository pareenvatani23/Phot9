/**
 * Environment configuration. All secrets are server-only and are never echoed
 * back in any client-visible response (see acceptance criterion #7).
 */

function required(name: string): string {
  const v = process.env[name];
  if (!v || v.trim() === "") {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return v;
}

export const config = {
  /** fal.ai API key — SERVER ONLY. */
  FAL_KEY: required("FAL_KEY"),
  PORT: Number(process.env.PORT ?? 8080),
  /** Base URL the iOS app uses to reach this service (for stable asset URLs). */
  PUBLIC_BASE_URL: (process.env.PUBLIC_BASE_URL ?? `http://localhost:${process.env.PORT ?? 8080}`).replace(/\/+$/, ""),
} as const;

/** Pipeline tunables (see spec sections 2.4 / 2.5). */
export const limits = {
  MAX_UPLOAD_BYTES: 12 * 1024 * 1024, // 12 MB (spec 2.3)
  PIPELINE_TIMEOUT_MS: 180_000, // 180 s (spec 2.5)
  FAL_POLL_INTERVAL_MS: 2_000, // 2 s (spec 2.5 / 4.3)
  FAL_RETRY_ATTEMPTS: 1, // retry once on failure (spec 2.5)
  BBOX_EXPAND: 0.08, // expand each person bbox by 8% (spec 2.4 Stage B)
  MIN_GLB_BYTES: 1024, // < 1 KB GLB => degenerate (spec 2.5)
} as const;
