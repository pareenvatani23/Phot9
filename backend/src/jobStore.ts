/**
 * In-memory job store (spec 2.1: a Map<jobId, JobRecord> is acceptable for v1).
 * No database. Jobs live for the lifetime of the process.
 */
import { randomUUID } from "node:crypto";
import type { DioramaResult, JobError, JobRecord, JobStage, JobStatus } from "./types.js";

const jobs = new Map<string, JobRecord>();

export function createJob(): JobRecord {
  const now = Date.now();
  const record: JobRecord = {
    id: randomUUID(),
    status: "queued",
    stage: "uploading",
    progress: 0,
    createdAt: now,
    updatedAt: now,
  };
  jobs.set(record.id, record);
  return record;
}

export function getJob(id: string): JobRecord | undefined {
  return jobs.get(id);
}

export function setStage(id: string, stage: JobStage, progress: number): void {
  const job = jobs.get(id);
  if (!job) return;
  job.status = "running";
  job.stage = stage;
  job.progress = progress;
  job.updatedAt = Date.now();
}

export function succeed(id: string, result: DioramaResult): void {
  const job = jobs.get(id);
  if (!job) return;
  job.status = "succeeded";
  job.progress = 1;
  job.result = result;
  job.updatedAt = Date.now();
}

export function fail(id: string, error: JobError): void {
  const job = jobs.get(id);
  if (!job) return;
  job.status = "failed";
  job.error = error;
  job.updatedAt = Date.now();
}

/** Shape returned by GET /v1/diorama/{id} — never leaks internal timestamps or secrets. */
export function toPublic(job: JobRecord): Record<string, unknown> {
  const base: Record<string, unknown> = { job_id: job.id, status: job.status };
  if (job.status === "running" || job.status === "queued") {
    base.stage = job.stage;
    base.progress = job.progress;
  }
  if (job.status === "succeeded") base.result = job.result;
  if (job.status === "failed") base.error = job.error;
  return base;
}

export type { JobStatus };
