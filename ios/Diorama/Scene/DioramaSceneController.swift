import SceneKit
import UIKit
import GLTFKit2

/// Builds the SceneKit scene from the packaged assets and drives the bounded
/// orbit camera (spec §3.3 / §3.4). The azimuth is HARD CLAMPED to a forward
/// arc so the user never swings into the unphotographed rear of the people.
///
/// The opening frame reproduces the ORIGINAL PHOTO: the camera FOV is derived
/// from SAM's focal length and the start distance from `avg_cam_tz`, so the
/// people appear at the same size/position as in the photo (not zoomed out).
/// When a depth map is supplied the background is a depth-displaced mesh
/// (navigable parallax) instead of a flat billboard.
final class DioramaSceneController: NSObject {

    // MARK: Orbit constants (spec §3.4)
    private let minAzimuth: Float = -1.221   // -70°
    private let maxAzimuth: Float =  1.221   //  +70°
    private let minElevation: Float = -0.20
    private let maxElevation: Float =  0.52
    private let rubberBandZone: Float = 0.15
    private let rubberBandFactor: Float = 0.35

    // MARK: State
    private(set) var azimuth: Float = 0       // 0 = front (original photo viewpoint)
    private var elevation: Float = 0
    private var orbitRadius: Float = 3
    private var minRadius: Float = 1.6
    private var maxRadius: Float = 4.0
    private var centroid = SCNVector3Zero     // people re-centered to origin
    private var worldOffset = SCNVector3Zero  // -original centroid; applied to splat to keep alignment
    private var cameraFOVDeg: CGFloat = 55    // vertical, derived from focal length

    let scnView: SCNView
    private let scene = SCNScene()
    private let cameraNode = SCNNode()
    private let peopleNode = SCNNode()

    private var lastPan = CGPoint.zero
    private var pinchStartRadius: Float = 3

    // MARK: Init

    /// - Parameters:
    ///   - heroGLB: local file URL of the (textured) hero GLB.
    ///   - backdropImage: backdrop photo texture.
    ///   - depthImage: optional monocular depth map of the photo → depth-mesh background.
    ///   - hint: backend hints (`focal_length`, `avg_cam_tz`, image size).
    init(heroGLB: URL, backdropImage: UIImage, depthImage: UIImage?, splatData: Data? = nil, hint: DioramaResult) throws {
        scnView = SCNView(frame: .zero)
        super.init()

        let imgW = hint.backdrop.img_w
        let imgH = hint.backdrop.img_h
        let focal = Float(hint.scene_hint.focal_length)

        // Vertical FOV of the original photo camera (pinhole): 2·atan((h/2)/f).
        if focal > 1, imgH > 0 {
            let fov = Double(2 * atan((Float(imgH) / 2) / focal) * 180 / .pi)
            cameraFOVDeg = CGFloat(min(90, max(25, fov)))
        }

        try buildPeople(from: heroGLB)

        // Start distance = real camera→subject distance, so the opening frame
        // matches the photo. Zoom range brackets it.
        let camDist = max(0.5, Float(hint.scene_hint.avg_cam_tz))
        orbitRadius = camDist
        minRadius = camDist * 0.45
        maxRadius = camDist * 2.2

        // Prefer the Gaussian-splat environment (navigable 3D); else depth-mesh / flat plane.
        if let splatData, let cloud = buildSplatCloud(from: splatData) {
            cloud.position = worldOffset   // align with the re-centered people
            scene.rootNode.addChildNode(cloud)
        } else {
            buildBackdrop(image: backdropImage, depth: depthImage, imgW: imgW, imgH: imgH)
        }
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

        // Re-center so the people centroid sits at world origin (orbit pivot).
        worldOffset = SCNVector3(-centroid.x, -centroid.y, -centroid.z)
        peopleNode.position = worldOffset
        centroid = SCNVector3Zero

        scene.rootNode.addChildNode(peopleNode)
    }

    // MARK: Backdrop (depth-mesh, or flat plane fallback)

    private func buildBackdrop(image: UIImage, depth: UIImage?, imgW: Int, imgH: Int) {
        let aspect = imgH > 0 ? Float(imgW) / Float(imgH) : 1
        // Sit the backdrop behind the people; far enough to read as background.
        let backdropDistance = orbitRadius          // plane center at z = -orbitRadius
        let camToPlane = orbitRadius + backdropDistance
        let fovRad = Float(cameraFOVDeg) * .pi / 180
        let planeHeight = 2 * camToPlane * tanf(fovRad / 2)
        let planeWidth = planeHeight * aspect

        if let depth = depth,
           let node = buildDepthMesh(image: image, depth: depth,
                                     width: planeWidth, height: planeHeight,
                                     baseZ: -backdropDistance,
                                     depthAmount: orbitRadius * 0.2) {
            scene.rootNode.addChildNode(node)
            return
        }

        // Flat billboard fallback.
        let plane = SCNPlane(width: CGFloat(planeWidth), height: CGFloat(planeHeight))
        let mat = SCNMaterial()
        mat.diffuse.contents = image
        mat.lightingModel = .constant
        mat.isDoubleSided = true
        plane.firstMaterial = mat
        let node = SCNNode(geometry: plane)
        node.position = SCNVector3(0, 0, -backdropDistance)
        scene.rootNode.addChildNode(node)
    }

    /// Tessellated grid textured with the photo and displaced along Z by the
    /// depth map (brighter = nearer, per Depth-Anything-style inverse depth), so
    /// the background has real parallax when the camera orbits.
    private func buildDepthMesh(image: UIImage, depth: UIImage, width: Float, height: Float,
                                baseZ: Float, depthAmount: Float) -> SCNNode? {
        guard let d = sampleGray(depth) else { return nil }
        let cols = 128
        let rows = max(2, Int((Float(cols) / max(0.1, width / height)).rounded()))

        var verts = [SCNVector3]()
        var uvs = [CGPoint]()
        verts.reserveCapacity(cols * rows)
        uvs.reserveCapacity(cols * rows)

        for j in 0..<rows {
            let v = Float(j) / Float(rows - 1)      // 0 at top
            for i in 0..<cols {
                let u = Float(i) / Float(cols - 1)
                let px = min(d.w - 1, Int(u * Float(d.w - 1)))
                let py = min(d.h - 1, Int(v * Float(d.h - 1)))
                let norm = Float(d.data[py * d.w + px]) / 255.0   // 1 = near, 0 = far
                let x = (u - 0.5) * width
                let y = (0.5 - v) * height
                let z = baseZ - (1 - norm) * depthAmount          // far pixels pushed back
                verts.append(SCNVector3(x, y, z))
                uvs.append(CGPoint(x: CGFloat(u), y: CGFloat(v)))      // upright (top vertex -> image top)
            }
        }

        var idx = [Int32]()
        idx.reserveCapacity((cols - 1) * (rows - 1) * 6)
        for j in 0..<(rows - 1) {
            for i in 0..<(cols - 1) {
                let a = Int32(j * cols + i)
                let b = a + 1
                let c = a + Int32(cols)
                let e = c + 1
                idx += [a, c, b, b, c, e]
            }
        }

        let vSource = SCNGeometrySource(vertices: verts)
        let tSource = SCNGeometrySource(textureCoordinates: uvs)
        let element = SCNGeometryElement(indices: idx, primitiveType: .triangles)
        let geo = SCNGeometry(sources: [vSource, tSource], elements: [element])
        let mat = SCNMaterial()
        mat.diffuse.contents = image
        mat.lightingModel = .constant
        mat.isDoubleSided = true
        geo.firstMaterial = mat
        return SCNNode(geometry: geo)
    }

    /// Decode an image to an 8-bit grayscale buffer.
    private func sampleGray(_ img: UIImage) -> (w: Int, h: Int, data: [UInt8])? {
        guard let cg = img.cgImage else { return nil }
        let w = cg.width, h = cg.height
        guard w > 0, h > 0 else { return nil }
        var data = [UInt8](repeating: 0, count: w * h)
        let cs = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(data: &data, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return (w, h, data)
    }

    // MARK: Gaussian-splat environment (rendered as a SceneKit point cloud)

    /// Parse the compact `.splat` records (32 bytes: pos 3×f32, scale 3×f32,
    /// color RGBA u8, rot 4×u8) into a colored SceneKit point cloud that
    /// composites with the people meshes under the shared orbit camera.
    private func buildSplatCloud(from data: Data) -> SCNNode? {
        let recordSize = 32
        let n = data.count / recordSize
        guard n > 0 else { return nil }

        var positions = [SCNVector3](); positions.reserveCapacity(n)
        var colors = [Float](); colors.reserveCapacity(n * 4)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            for i in 0..<n {
                let base = i * recordSize
                let x = raw.loadUnaligned(fromByteOffset: base, as: Float32.self)
                let y = raw.loadUnaligned(fromByteOffset: base + 4, as: Float32.self)
                let z = raw.loadUnaligned(fromByteOffset: base + 8, as: Float32.self)
                positions.append(SCNVector3(x, y, z))
                colors.append(Float(raw[base + 24]) / 255)
                colors.append(Float(raw[base + 25]) / 255)
                colors.append(Float(raw[base + 26]) / 255)
                colors.append(Float(raw[base + 27]) / 255)
            }
        }

        let vSource = SCNGeometrySource(vertices: positions)
        let colorData = colors.withUnsafeBytes { Data($0) }
        let cSource = SCNGeometrySource(
            data: colorData, semantic: .color, vectorCount: n,
            usesFloatComponents: true, componentsPerVector: 4,
            bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
            dataStride: MemoryLayout<Float>.size * 4)

        let indices = [Int32](0..<Int32(n))
        let element = SCNGeometryElement(indices: indices, primitiveType: .point)
        element.minimumPointScreenSpaceRadius = 4
        element.maximumPointScreenSpaceRadius = 14
        element.pointSize = 10

        let geo = SCNGeometry(sources: [vSource, cSource], elements: [element])
        let mat = SCNMaterial()
        mat.lightingModel = .constant
        // Soft radial-falloff sprite so each point reads like a Gaussian blob
        // (modulated by the per-vertex photo color) instead of a hard square.
        mat.diffuse.contents = softPointSprite()
        mat.isDoubleSided = true
        geo.firstMaterial = mat
        return SCNNode(geometry: geo)
    }

    /// White radial gradient (opaque center → transparent edge) used as a point
    /// sprite to approximate Gaussian falloff.
    private func softPointSprite(size: CGFloat = 64) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: size, height: size))
        return renderer.image { ctx in
            let cg = ctx.cgContext
            let colors = [UIColor(white: 1, alpha: 1).cgColor, UIColor(white: 1, alpha: 0).cgColor] as CFArray
            guard let grad = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(),
                                        colors: colors, locations: [0, 1]) else { return }
            let c = CGPoint(x: size / 2, y: size / 2)
            cg.drawRadialGradient(grad, startCenter: c, startRadius: 0,
                                  endCenter: c, endRadius: size / 2, options: [])
        }
    }

    // MARK: Lighting + ground shadow

    private func buildLightingAndFloor() {
        let ambient = SCNLight()
        ambient.type = .ambient
        ambient.intensity = 500
        let ambientNode = SCNNode()
        ambientNode.light = ambient
        scene.rootNode.addChildNode(ambientNode)

        let directional = SCNLight()
        directional.type = .directional
        directional.intensity = 650
        let dirNode = SCNNode()
        dirNode.light = directional
        dirNode.eulerAngles = SCNVector3(-Float.pi / 3, Float.pi / 6, 0) // front-top
        scene.rootNode.addChildNode(dirNode)
    }

    // MARK: Camera

    private func buildCamera() {
        let camera = SCNCamera()
        camera.fieldOfView = cameraFOVDeg
        camera.projectionDirection = .vertical
        camera.zNear = 0.05
        camera.zFar = 5000
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
        elevation = 0
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
