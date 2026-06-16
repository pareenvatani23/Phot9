/**
 * fal.ai client wrapper. Implements the submit + poll + result pattern
 * (spec 4.3) with a 2 s poll interval and an overall deadline. The FAL_KEY is
 * configured here and never leaves the server.
 */
import { fal } from "@fal-ai/client";
import { config, limits } from "./config.js";
import { log } from "./logger.js";

fal.config({ credentials: config.FAL_KEY });

/** Upload a buffer to fal storage and return its public URL (spec 4.3). */
export async function uploadToFal(buffer: Buffer, contentType: string, fileName: string): Promise<string> {
  // The fal client accepts a web Blob/File; Blob is global in Node 20+.
  const blob = new Blob([buffer], { type: contentType });
  // Attach a name so fal preserves the extension.
  const file = new File([blob], fileName, { type: contentType });
  return fal.storage.upload(file);
}

export interface RunOptions {
  /** Absolute deadline (Date.now() ms). Throws "TIMEOUT" if exceeded. */
  deadline: number;
}

/**
 * Submit a fal job, poll until terminal, and return its typed result data.
 * Throws on FAILED/error or when the deadline passes.
 */
export async function runFal<T>(endpoint: string, input: Record<string, unknown>, opts: RunOptions): Promise<T> {
  const { request_id } = await fal.queue.submit(endpoint, { input });
  log.info("fal submitted", { endpoint, request_id });

  for (;;) {
    if (Date.now() > opts.deadline) {
      throw new Error("TIMEOUT");
    }
    const status = await fal.queue.status(endpoint, { requestId: request_id, logs: false });
    const state: string = status.status;
    if (state === "COMPLETED") {
      const out = await fal.queue.result(endpoint, { requestId: request_id });
      return out.data as T;
    }
    // Anything that isn't IN_QUEUE / IN_PROGRESS is treated as a failure.
    if (state !== "IN_QUEUE" && state !== "IN_PROGRESS") {
      throw new Error(`fal ${endpoint} returned status ${state}`);
    }
    await sleep(limits.FAL_POLL_INTERVAL_MS);
  }
}

/**
 * Run a fal job with one retry on failure (spec 2.5). Timeouts are NOT
 * retried — they propagate so the pipeline can fail the job as TIMEOUT.
 */
export async function runFalWithRetry<T>(
  endpoint: string,
  input: Record<string, unknown>,
  opts: RunOptions
): Promise<T> {
  let lastErr: unknown;
  for (let attempt = 0; attempt <= limits.FAL_RETRY_ATTEMPTS; attempt++) {
    try {
      return await runFal<T>(endpoint, input, opts);
    } catch (err) {
      if (err instanceof Error && err.message === "TIMEOUT") throw err;
      lastErr = err;
      log.warn("fal call failed", { endpoint, attempt, error: errMessage(err) });
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
}

export function errMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
