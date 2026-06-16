/**
 * Shared types: job model + the fal API contracts (copy-faithful from the
 * live fal schema, spec section 4) + the result schema returned to the app
 * (spec section 2.6).
 */

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export type JobStage =
  | "uploading"
  | "segmenting"
  | "reconstructing_bodies"
  | "building_backdrop"
  | "aligning"
  | "packaging";

export type ErrorCode =
  | "NO_PEOPLE_DETECTED"
  | "RECON_FAILED"
  | "TIMEOUT"
  | "BAD_INPUT"
  | "INTERNAL";

export interface JobError {
  code: ErrorCode;
  message: string;
}

/** Result object returned to the app (spec 2.6). */
export interface DioramaResult {
  aligned: boolean;
  hero_glb_url: string;
  backdrop: {
    image_url: string;
    img_w: number;
    img_h: number;
  };
  scene_hint: {
    num_people: number;
    avg_cam_tz: number;
    focal_length: number;
    people_bboxes: number[][]; // [x_min, y_min, x_max, y_max] per person
  };
}

export interface JobRecord {
  id: string;
  status: JobStatus;
  stage: JobStage;
  progress: number; // 0..1
  result?: DioramaResult;
  error?: JobError;
  createdAt: number;
  updatedAt: number;
}

// ---------------------------------------------------------------------------
// fal: fal-ai/sam-3/3d-body (spec 4.1)
// ---------------------------------------------------------------------------

export interface FalFile {
  url: string;
  content_type?: string;
  file_name?: string;
  file_size?: number;
}

export interface Sam3BodyPerson {
  person_id: number;
  bbox: [number, number, number, number]; // [x_min, y_min, x_max, y_max] px
  focal_length: number;
  pred_cam_t: [number, number, number]; // [tx, ty, tz]
  // keypoints_2d / keypoints_3d / MHR params present but unused in v1.
}

export interface Sam3BodyMetadata {
  num_people: number;
  keypoint_names?: string[];
  people: Sam3BodyPerson[];
}

export interface Sam3BodyOutput {
  model_glb: FalFile; // combined multi-person GLB (hero asset)
  visualization?: FalFile;
  meshes?: FalFile[];
  metadata: Sam3BodyMetadata;
}

// ---------------------------------------------------------------------------
// fal: fal-ai/sam-3/3d-align (spec 4.2)
// ---------------------------------------------------------------------------

export interface Sam3AlignMetadata {
  person_id?: number;
  scale_factor: number;
  translation: [number, number, number]; // [tx, ty, tz]
  focal_length?: number;
  target_points_count?: number;
  cropped_vertices_count?: number;
}

export interface Sam3AlignOutput {
  body_mesh_ply?: FalFile;
  model_glb: FalFile; // aligned GLB (hero asset when align succeeds)
  visualization?: FalFile;
  metadata: Sam3AlignMetadata;
  scene_glb?: FalFile; // only when object_mesh_url provided (absent in v1)
}
