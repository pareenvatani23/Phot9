import Foundation

/// Thin client for the Diorama backend (spec §2.3). Uploads a still, polls the
/// job, and downloads the packaged assets. No fal/credentials here — the app
/// only ever talks to our own backend.
struct DioramaAPI {
    let baseURL: URL
    private let session: URLSession

    init(baseURL: URL = AppConfig.backendBaseURL, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.session = session
    }

    enum ClientError: LocalizedError {
        case badResponse
        case server(APIError)
        case timedOut

        var errorDescription: String? {
            switch self {
            case .badResponse: return "Unexpected response from the server."
            case .timedOut: return "Building your diorama took too long."
            case .server(let e): return e.message
            }
        }
    }

    // MARK: - Upload

    /// POST a JPEG to `/v1/diorama`. Returns the created job id.
    func submit(jpeg: Data) async throws -> String {
        let url = baseURL.appendingPathComponent("v1/diorama")
        let boundary = "Boundary-\(UUID().uuidString)"
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        req.httpBody = Self.multipartBody(boundary: boundary, field: "photo", filename: "capture.jpg", mimeType: "image/jpeg", data: jpeg)

        let (data, response) = try await session.data(for: req)
        guard let http = response as? HTTPURLResponse else { throw ClientError.badResponse }
        guard http.statusCode == 202 else { throw try Self.decodeError(data) }
        return try JSONDecoder().decode(CreateJobResponse.self, from: data).job_id
    }

    // MARK: - Poll

    /// One status read of `/v1/diorama/{id}`.
    func status(jobId: String) async throws -> JobStatusResponse {
        let url = baseURL.appendingPathComponent("v1/diorama/\(jobId)")
        let (data, response) = try await session.data(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw try Self.decodeError(data)
        }
        return try JSONDecoder().decode(JobStatusResponse.self, from: data)
    }

    /// Poll every `AppConfig.pollInterval` until the job succeeds or fails.
    /// Calls `onStage` with the advisory stage label for the progress UI.
    func awaitResult(jobId: String, onStage: @escaping (String?, Double?) -> Void) async throws -> DioramaResult {
        while true {
            let s = try await status(jobId: jobId)
            switch s.status {
            case .queued, .running:
                onStage(s.stage, s.progress)
            case .succeeded:
                guard let result = s.result else { throw ClientError.badResponse }
                return result
            case .failed:
                throw ClientError.server(s.error ?? APIError(code: "RECON_FAILED", message: "Something went wrong."))
            }
            try await Task.sleep(nanoseconds: UInt64(AppConfig.pollInterval * 1_000_000_000))
        }
    }

    // MARK: - Download

    /// Download an asset (hero GLB or backdrop) to a local file URL.
    func download(_ remote: URL, suggestedName: String) async throws -> URL {
        let (tmp, response) = try await session.download(from: remote)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw ClientError.badResponse
        }
        let dest = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString)-\(suggestedName)")
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.moveItem(at: tmp, to: dest)
        return dest
    }

    // MARK: - Helpers

    private static func multipartBody(boundary: String, field: String, filename: String, mimeType: String, data: Data) -> Data {
        var body = Data()
        func append(_ s: String) { body.append(s.data(using: .utf8)!) }
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"\(field)\"; filename=\"\(filename)\"\r\n")
        append("Content-Type: \(mimeType)\r\n\r\n")
        body.append(data)
        append("\r\n--\(boundary)--\r\n")
        return body
    }

    private static func decodeError(_ data: Data) throws -> ClientError {
        struct Wrapper: Codable { let error: APIError }
        if let w = try? JSONDecoder().decode(Wrapper.self, from: data) {
            return .server(w.error)
        }
        return .badResponse
    }
}
