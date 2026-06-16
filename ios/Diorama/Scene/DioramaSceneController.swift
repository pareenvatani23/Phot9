import SceneKit
import UIKit
import GLTFKit2

/// Builds the SceneKit scene from the packaged assets and drives the bounded
/// orbit camera (spec §3.3 / §3.4). The azimuth is HARD CLAMPED to a forward
/// arc so the user never swings into the unphotographed rear of the people.
final class DioramaSceneController: NSObject {

    // MARK: Orbit constants (spec §3.4)
    private let minAzimuth: Float = -1.221   // -70°
    private let maxAzimuth: Float =  1.221   //  +70°
    private let minElevation: Float = -0.20
    private let maxElevation: Float =  0.52
    private let cameraFOV: CGFloat = 55       // vertical, degrees
    private let rubberBandZone: Float = 0.15
    private let rubberBandFactor: Float = 0.35

    // MARK: State
    private(set) var azimuth: Float = 0       // 0 = front (original photo viewpoint)
    private var elevation: Float = 0.15
    private var orbitRadius: Float = 3
    private var minRadius: Float = 1.6
    private var maxRadius: Float = 4.0
    private var centroid = SCNVector3Zero     // people re-centered to origin

    let scnView: SCNView
    private let scene = SCNScene()
    private let cameraNode = SCNNode()
    private let peopleNode = SCNNode()

    private var lastPan = CGPoint.zero
    private var pinchStartRadius: Float = 3

    // MARK: Init

    /// - Parameters:
    ///   - heroGLB: local file URL of the hero GLB.
    ///   - backdropImage: backdrop texture.
    ///   - hint: scene hints from the backend (`avg_cam_tz`, aspect, etc.).
    init(heroGLB: URL, backdropImage: UIImage, hint: DioramaResult) throws {
        scnView = SCNView(frame: .zero)
        super.init()

        try buildPeople(from: heroGLB)
        buildBackdrop(image: backdropImage,
                      imgW: hint.backdrop.img_w,
                      imgH: hint.backdrop.img_h,
                      avgCamTz: Float(hint.scene_hint.avg_cam_tz))
        buildLightingAndFloor()
        buildCamera()

        scnView.scene = scene
        scnView.backgroundColor = .black
        scnView.antialiasingMode = .multisampling4X
        scnView.allowsCameraControl = false   // custom orbit only — never the default

        installGestures()
        updateCamera()
    }

    // MARK: People (hero GLB)

    private func buildPeople(from url: URL) throws {
        let asset = try GLTFAsset(url: url)
        let source = GLTFSCNSceneSource(asset: asset)
        guard let loaded = source.defaultScene ?? source.scenes.first else {
            throw NSError(domain: "Diorama", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "Couldn't load the 3D scene."])
        }
        for child in loaded.rootNode.childNodes {
            peopleNode.addChildNode(child)
        }

        // Combined bounding box across all descendant geometry (spec §3.3).
        let (minV, maxV) = combinedBoundingBox(of: peopleNode) ?? (SCNVector3Zero, SCNVector3Zero)
        centroid = SCNVector3((minV.x + maxV.x) / 2, (minV.y + maxV.y) / 2, (minV.z + maxV.z) / 2)
        let dx = maxV.x - minV.x, dy = maxV.y - minV.y, dz = maxV.z - minV.z
        let radius = max(0.5, 0.5 * sqrtf(dx * dx + dy * dy + dz * dz))

        // Re-center so the people centroid sits at world origin.
        peopleNode.position = SCNVector3(-centroid.x, -centroid.y, -centroid.z)
        centroid = SCNVector3Zero

        // Seed orbit radius (spec §3.4): radius * 2.5, clamped to [1.6r, 4r].
        minRadius = radius * 1.6
        maxRadius = radius * 4.0
        orbitRadius = min(maxRadius, max(minRadius, radius * 2.5))

        scene.rootNode.addChildNode(peopleNode)
    }

    // MARK: Backdrop (camera-facing plane)

    private func buildBackdrop(image: UIImage, imgW: Int, imgH: Int, avgCamTz: Float) {
        let backdropDistance = max(2.0, avgCamTz * 1.4)   // people sit in front of it (spec §3.3)

        // Distance from the home camera (on +Z at orbitRadius) to the plane.
        let camToPlane = orbitRadius + backdropDistance
        let fovRad = Float(cameraFOV) * .pi / 180
        let planeHeight = 2 * camToPlane * tanf(fovRad / 2)
        let aspect = imgH > 0 ? Float(imgW) / Float(imgH) : 1
        let planeWidth = planeHeight * aspect

        let plane = SCNPlane(width: CGFloat(planeWidth), height: CGFloat(planeHeight))
        let mat = SCNMaterial()
        mat.diffuse.contents = image
        mat.lightingModel = .constant      // backdrop is fully lit (a photo), not shaded
        mat.isDoubleSided = true
        plane.firstMaterial = mat

        let node = SCNNode(geometry: plane)
        // Fixed billboard at the rear; it does NOT follow the camera so parallax
        // reads correctly between people and background (spec §3.3).
        node.position = SCNVector3(0, 0, -backdropDistance)
        scene.rootNode.addChildNode(node)
    }

    // MARK: Lighting + ground shadow

    private func buildLightingAndFloor() {
        let ambient = SCNLight()
        ambient.type = .ambient
        ambient.intensity = 400
        let ambientNode = SCNNode()
        ambientNode.light = ambient
        scene.rootNode.addChildNode(ambientNode)

        let directional = SCNLight()
        directional.type = .directional
        directional.intensity = 700
        directional.castsShadow = true
        directional.shadowMode = .deferred
        let dirNode = SCNNode()
        dirNode.light = directional
        dirNode.eulerAngles = SCNVector3(-Float.pi / 3, Float.pi / 6, 0) // front-top
        scene.rootNode.addChildNode(dirNode)

        // Soft ground shadow so the people read as standing in a space.
        let floor = SCNFloor()
        floor.reflectivity = 0
        let floorMat = SCNMaterial()
        floorMat.diffuse.contents = UIColor(white: 0.05, alpha: 1)
        floorMat.lightingModel = .lambert
        floor.firstMaterial = floorMat
        let floorNode = SCNNode(geometry: floor)
        // Drop the floor to the feet: bottom of the (re-centered) people bbox.
        let (minV, _) = combinedBoundingBox(of: peopleNode) ?? (SCNVector3Zero, SCNVector3Zero)
        floorNode.position = SCNVector3(0, minV.y, 0)
        scene.rootNode.addChildNode(floorNode)
    }

    // MARK: Camera

    private func buildCamera() {
        let camera = SCNCamera()
        camera.fieldOfView = cameraFOV
        camera.projectionDirection = .vertical
        camera.zNear = 0.05
        camera.zFar = 1000
        cameraNode.camera = camera
        scene.rootNode.addChildNode(cameraNode)
    }

    /// Recompute camera position from (azimuth, elevation, orbitRadius) — spec §3.4.
    func updateCamera() {
        let r = orbitRadius
        let x = centroid.x + r * sin(azimuth) * cos(elevation)
        let y = centroid.y + r * sin(elevation)
        let z = centroid.z + r * cos(azimuth) * cos(elevation)
        cameraNode.position = SCNVector3(x, y, z)
        cameraNode.look(at: centroid, up: SCNVector3(0, 1, 0), localFront: SCNVector3(0, 0, -1))
    }

    // MARK: Gestures

    private func installGestures() {
        let pan = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        let pinch = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        scnView.addGestureRecognizer(pan)
        scnView.addGestureRecognizer(pinch)
    }

    @objc private func handlePan(_ gr: UIPanGestureRecognizer) {
        let t = gr.translation(in: scnView)
        if gr.state == .began { lastPan = .zero }
        let dxTranslation = t.x - lastPan.x
        let dyTranslation = t.y - lastPan.y
        lastPan = t

        var dAz = Float(-dxTranslation) * 0.005
        var dEl = Float(-dyTranslation) * 0.005

        // Rubber-band: resist near the azimuth clamp instead of stopping dead.
        if (azimuth > maxAzimuth - rubberBandZone && dAz > 0) ||
           (azimuth < minAzimuth + rubberBandZone && dAz < 0) {
            dAz *= rubberBandFactor
        }
        if (elevation > maxElevation - rubberBandZone && dEl > 0) ||
           (elevation < minElevation + rubberBandZone && dEl < 0) {
            dEl *= rubberBandFactor
        }

        azimuth = min(maxAzimuth, max(minAzimuth, azimuth + dAz))      // HARD CLAMP — the void guard
        elevation = min(maxElevation, max(minElevation, elevation + dEl))
        updateCamera()
    }

    @objc private func handlePinch(_ gr: UIPinchGestureRecognizer) {
        if gr.state == .began { pinchStartRadius = orbitRadius }
        let proposed = pinchStartRadius / Float(gr.scale)
        orbitRadius = min(maxRadius, max(minRadius, proposed))
        updateCamera()
    }

    // MARK: Programmatic control (used by the recorder)

    func setAzimuth(_ value: Float) {
        azimuth = min(maxAzimuth, max(minAzimuth, value))
        updateCamera()
    }

    func resetToFront() {
        azimuth = 0
        elevation = 0.15
        updateCamera()
    }

    // MARK: Continuous auto-orbit (demo / preview)

    private var autoLink: CADisplayLink?
    private var autoStart: CFTimeInterval = 0
    private var autoPeriod: Double = 6

    /// Smoothly sweep the azimuth back and forth (±0.9 rad) forever, so an
    /// on-screen recording shows the parallax. Drives the live SCNView.
    func startAutoOrbit(period: Double = 6) {
        stopAutoOrbit()
        autoPeriod = period
        autoStart = CACurrentMediaTime()
        scnView.rendersContinuously = true
        let link = CADisplayLink(target: self, selector: #selector(stepAutoOrbit))
        link.add(to: .main, forMode: .common)
        autoLink = link
    }

    func stopAutoOrbit() {
        autoLink?.invalidate()
        autoLink = nil
        scnView.rendersContinuously = false
    }

    @objc private func stepAutoOrbit() {
        let t = CACurrentMediaTime() - autoStart
        let phase = sin(2 * Double.pi * t / autoPeriod) // -1...1
        setAzimuth(Float(phase) * 0.9)
    }

    // MARK: Bounding box helper

    /// Combined AABB of all descendant geometry, expressed in `root`'s space.
    private func combinedBoundingBox(of root: SCNNode) -> (min: SCNVector3, max: SCNVector3)? {
        var minV = SCNVector3(Float.greatestFiniteMagnitude, .greatestFiniteMagnitude, .greatestFiniteMagnitude)
        var maxV = SCNVector3(-Float.greatestFiniteMagnitude, -.greatestFiniteMagnitude, -.greatestFiniteMagnitude)
        var found = false

        func visit(_ node: SCNNode) {
            if node.geometry != nil {
                let (lo, hi) = node.boundingBox
                let corners = [
                    SCNVector3(lo.x, lo.y, lo.z), SCNVector3(hi.x, lo.y, lo.z),
                    SCNVector3(lo.x, hi.y, lo.z), SCNVector3(lo.x, lo.y, hi.z),
                    SCNVector3(hi.x, hi.y, lo.z), SCNVector3(hi.x, lo.y, hi.z),
                    SCNVector3(lo.x, hi.y, hi.z), SCNVector3(hi.x, hi.y, hi.z),
                ]
                for c in corners {
                    let p = root.convertPosition(c, from: node)
                    minV = SCNVector3(min(minV.x, p.x), min(minV.y, p.y), min(minV.z, p.z))
                    maxV = SCNVector3(max(maxV.x, p.x), max(maxV.y, p.y), max(maxV.z, p.z))
                    found = true
                }
            }
            for child in node.childNodes { visit(child) }
        }
        visit(root)
        return found ? (minV, maxV) : nil
    }
}
