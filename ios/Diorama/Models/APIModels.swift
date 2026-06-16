import Foundation

/// Job lifecycle states (spec §2.3).
enum JobStatus: String, Codable {
    case queued, running, succeeded, failed
}

/// Response of `POST /v1/diorama`.
struct CreateJobResponse: Codable {
    let job_id: String
    let status: JobStatus
}

/// Backend error payload (spec §2.3 / §3.6).
struct APIError: Codable, Error {
    let code: String
    let message: String
}

/// The hero result (spec §2.6).
struct DioramaResult: Codable {
    struct Backdrop: Codable {
        let image_url: String
        let img_w: Int
        let img_h: Int
    }
    struct SceneHint: Codable {
        let num_people: Int
        let avg_cam_tz: Double
        let focal_length: Double
        let people_bboxes: [[Double]]
    }
    let aligned: Bool
    let hero_glb_url: String
    let backdrop: Backdrop
    let scene_hint: SceneHint
}

/// Response of `GET /v1/diorama/{id}` — fields are conditional on `status`.
struct JobStatusResponse: Codable {
    let job_id: String
    let status: JobStatus
    let stage: String?
    let progress: Double?
    let result: DioramaResult?
    let error: APIError?
}
