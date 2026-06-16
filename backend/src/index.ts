/**
 * HTTP layer (spec 2.3): the thin backend proxy. Holds the FAL_KEY, exposes
 * the two job endpoints, hosts packaged assets, and kicks off the pipeline.
 */
import express, { type Request, type Response } from "express";
import multer from "multer";
import { config, limits } from "./config.js";
import { createJob, getJob, toPublic } from "./jobStore.js";
import { log } from "./logger.js";
import { ASSET_ROOT, runPipeline } from "./pipeline.js";

const app = express();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: limits.MAX_UPLOAD_BYTES },
  fileFilter: (_req, file, cb) => {
    const ok = file.mimetype === "image/jpeg" || file.mimetype === "image/png";
    if (ok) cb(null, true);
    else cb(new Error("Only JPEG or PNG is accepted"));
  },
});

app.get("/health", (_req: Request, res: Response) => {
  res.json({ ok: true });
});

// Stable asset hosting (spec 2.4 Stage D / 2.6): hero.glb + backdrop.jpg.
app.use("/assets", express.static(ASSET_ROOT, { fallthrough: false, maxAge: "1h" }));

// POST /v1/diorama — accept a photo, create a job, kick off the pipeline (202).
app.post("/v1/diorama", upload.single("photo"), (req: Request, res: Response) => {
  if (!req.file) {
    res.status(400).json({ error: { code: "BAD_INPUT", message: "Missing 'photo' file field" } });
    return;
  }
  const job = createJob();
  log.info("job created", { jobId: job.id, bytes: req.file.size, type: req.file.mimetype });

  // Fire-and-forget; progress is tracked in the job store.
  void runPipeline(job.id, req.file.buffer, req.file.mimetype);

  res.status(202).json({ job_id: job.id, status: "queued" });
});

// GET /v1/diorama/:id — current job state.
app.get("/v1/diorama/:id", (req: Request, res: Response) => {
  const job = getJob(req.params.id);
  if (!job) {
    res.status(404).json({ error: { code: "NOT_FOUND", message: "Unknown job id" } });
    return;
  }
  res.status(200).json(toPublic(job));
});

// Multer / payload errors -> clean JSON.
app.use((err: unknown, _req: Request, res: Response, _next: express.NextFunction) => {
  const message = err instanceof Error ? err.message : "Upload error";
  const code = message.includes("File too large") ? "PAYLOAD_TOO_LARGE" : "BAD_INPUT";
  log.warn("request rejected", { code, message });
  res.status(400).json({ error: { code, message } });
});

app.listen(config.PORT, () => {
  log.info(`Diorama backend listening on :${config.PORT}`, { publicBaseUrl: config.PUBLIC_BASE_URL });
});
