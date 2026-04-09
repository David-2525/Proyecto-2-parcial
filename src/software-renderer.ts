import { clamp, mat4, vec2, vec3, type Mat4, type Vec2, type Vec3 } from "./math";
import type { IndexedMesh } from "./mesh";

export type ShadingMode = "gouraud" | "phong" | "normals" | "wireframe";

export interface TextureData {
  width: number;
  height: number;
  pixels: Uint8ClampedArray;
}

export interface LightSettings {
  position: Vec3;
  ambient: number;
  diffuse: number;
  specular: number;
  specularPower: number;
  shadowEnabled: boolean;
  shadowStrength: number;
  shadowBias: number;
}

export interface RenderScene {
  objects: RenderObject[];
  viewMatrix: Mat4;
  projectionMatrix: Mat4;
  cameraPosition: Vec3;
  near: number;
  far: number;
  texture: TextureData;
  shading: ShadingMode;
  light: LightSettings;
  shadowCenter: Vec3;
  shadowRadius: number;
  clearColor: [number, number, number];
  resolutionScale: number;
  maxPixels: number;
  shadowMapSize: number;
  selectedObjectId: number | null;
}

export interface RenderObject {
  id: number;
  mesh: IndexedMesh;
  modelMatrix: Mat4;
}

export interface RenderStats {
  width: number;
  height: number;
  submittedTriangles: number;
  rasterizedTriangles: number;
  renderTimeMs: number;
}

interface ClipVertex {
  viewPosition: Vec3;
  worldPosition: Vec3;
  normal: Vec3;
  uv: Vec2;
}

interface ProjectedVertex extends ClipVertex {
  screenX: number;
  screenY: number;
  invW: number;
  depth: number;
  gouraud: Vec3;
}

interface ShadowCamera {
  viewMatrix: Mat4;
  projectionMatrix: Mat4;
  depthBuffer: Float32Array;
  size: number;
}

export class SoftwareRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private canvasWidth = 1;
  private canvasHeight = 1;
  private imageData = new ImageData(1, 1);
  private colorData = this.imageData.data;
  private depthBuffer = new Float32Array(1);
  private positionBuffer = new Float32Array(3);
  private normalBuffer = new Float32Array(3);
  private albedoBuffer = new Float32Array(3);
  private gouraudBuffer = new Float32Array(3);
  private edgeMask = new Uint8Array(1);
  private coverageMask = new Uint8Array(1);
  private objectIdBuffer = new Int32Array(1);
  private shadowMapSize = 1;
  private shadowDepthBuffer = new Float32Array(1);

  get width() {
    return this.canvasWidth;
  }

  get height() {
    return this.canvasHeight;
  }

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;

    const context = canvas.getContext("2d");

    if (!context) {
      throw new Error("2D canvas context is not available");
    }

    this.ctx = context;
    this.resize();
  }

  resize(resolutionScale = 1, maxPixels = 540_000) {
    const clientWidth = Math.max(320, Math.floor(this.canvas.clientWidth || window.innerWidth));
    const clientHeight = Math.max(240, Math.floor(this.canvas.clientHeight || window.innerHeight));
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.1);
    let width = Math.floor(clientWidth * pixelRatio * resolutionScale);
    let height = Math.floor(clientHeight * pixelRatio * resolutionScale);

    width = Math.max(280, width);
    height = Math.max(210, height);

    if (width * height > maxPixels) {
      const scale = Math.sqrt(maxPixels / (width * height));
      width = Math.max(280, Math.floor(width * scale));
      height = Math.max(210, Math.floor(height * scale));
    }

    if (width === this.canvasWidth && height === this.canvasHeight) {
      return;
    }

    this.canvasWidth = width;
    this.canvasHeight = height;
    this.canvas.width = width;
    this.canvas.height = height;

    this.imageData = new ImageData(width, height);
    this.colorData = this.imageData.data;
    this.depthBuffer = new Float32Array(width * height);
    this.positionBuffer = new Float32Array(width * height * 3);
    this.normalBuffer = new Float32Array(width * height * 3);
    this.albedoBuffer = new Float32Array(width * height * 3);
    this.gouraudBuffer = new Float32Array(width * height * 3);
    this.edgeMask = new Uint8Array(width * height);
    this.coverageMask = new Uint8Array(width * height);
    this.objectIdBuffer = new Int32Array(width * height);
  }

  private ensureShadowMap(size: number) {
    const nextSize = Math.max(64, Math.floor(size));

    if (nextSize === this.shadowMapSize) {
      return;
    }

    this.shadowMapSize = nextSize;
    this.shadowDepthBuffer = new Float32Array(nextSize * nextSize);
  }

  render(scene: RenderScene): RenderStats {
    const start = performance.now();
    this.resize(scene.resolutionScale, scene.maxPixels);
    this.clear(scene.clearColor);

    let rasterizedTriangles = 0;
    let submittedTriangles = 0;

    const preparedObjects = scene.objects.map((renderObject) => {
      const modelView = mat4.multiply(scene.viewMatrix, renderObject.modelMatrix);
      const positionsView = new Float32Array(renderObject.mesh.vertexCount * 3);
      const positionsWorld = new Float32Array(renderObject.mesh.vertexCount * 3);
      const normalsWorld = new Float32Array(renderObject.mesh.vertexCount * 3);

      for (let vertex = 0; vertex < renderObject.mesh.vertexCount; vertex += 1) {
        const base = vertex * 3;
        const objectPosition: Vec3 = [
          renderObject.mesh.positions[base],
          renderObject.mesh.positions[base + 1],
          renderObject.mesh.positions[base + 2],
        ];
        const objectNormal: Vec3 = [
          renderObject.mesh.normals[base],
          renderObject.mesh.normals[base + 1],
          renderObject.mesh.normals[base + 2],
        ];

        const worldPosition = mat4.transformPoint(renderObject.modelMatrix, objectPosition);
        const viewPosition = mat4.transformPoint(modelView, objectPosition);
        const worldNormal = vec3.normalize(mat4.transformVector(renderObject.modelMatrix, objectNormal));

        positionsView[base] = viewPosition[0];
        positionsView[base + 1] = viewPosition[1];
        positionsView[base + 2] = viewPosition[2];

        positionsWorld[base] = worldPosition[0];
        positionsWorld[base + 1] = worldPosition[1];
        positionsWorld[base + 2] = worldPosition[2];

        normalsWorld[base] = worldNormal[0];
        normalsWorld[base + 1] = worldNormal[1];
        normalsWorld[base + 2] = worldNormal[2];
      }

      return {
        id: renderObject.id,
        mesh: renderObject.mesh,
        positionsView,
        positionsWorld,
        normalsWorld,
      };
    });

    const needsLighting = scene.shading === "gouraud" || scene.shading === "phong";
    const shadowCamera =
      needsLighting && scene.light.shadowEnabled ? this.createShadowCamera(scene) : null;

    if (shadowCamera) {
      shadowCamera.depthBuffer.fill(Number.POSITIVE_INFINITY);

      for (const renderObject of preparedObjects) {
        this.rasterizeShadowMesh(renderObject.mesh, renderObject.positionsWorld, shadowCamera);
      }
    }

    for (const renderObject of preparedObjects) {
      submittedTriangles += renderObject.mesh.triangleCount;

      for (let triangle = 0; triangle < renderObject.mesh.triangleCount; triangle += 1) {
        const base = triangle * 3;
        const ia = renderObject.mesh.indices[base];
        const ib = renderObject.mesh.indices[base + 1];
        const ic = renderObject.mesh.indices[base + 2];

        const a = this.makeClipVertex(
          ia,
          renderObject.positionsView,
          renderObject.positionsWorld,
          renderObject.normalsWorld,
          renderObject.mesh.uvs,
        );
        const b = this.makeClipVertex(
          ib,
          renderObject.positionsView,
          renderObject.positionsWorld,
          renderObject.normalsWorld,
          renderObject.mesh.uvs,
        );
        const c = this.makeClipVertex(
          ic,
          renderObject.positionsView,
          renderObject.positionsWorld,
          renderObject.normalsWorld,
          renderObject.mesh.uvs,
        );

        if (isBackFacing(a.viewPosition, b.viewPosition, c.viewPosition)) {
          continue;
        }

        const triangles = this.clipTriangle(this.unwrapTriangleU([a, b, c]), scene.near, scene.far);

        for (const clippedTriangle of triangles) {
          this.rasterizeTriangle(clippedTriangle, scene, renderObject.id, shadowCamera);
          rasterizedTriangles += 1;
        }
      }
    }

    this.compose(scene, shadowCamera);
    this.ctx.putImageData(this.imageData, 0, 0);

    return {
      width: this.canvasWidth,
      height: this.canvasHeight,
      submittedTriangles,
      rasterizedTriangles,
      renderTimeMs: performance.now() - start,
    };
  }

  pick(clientX: number, clientY: number) {
    const rect = this.canvas.getBoundingClientRect();
    const normalizedX = (clientX - rect.left) / rect.width;
    const normalizedY = (clientY - rect.top) / rect.height;

    if (normalizedX < 0 || normalizedX > 1 || normalizedY < 0 || normalizedY > 1) {
      return -1;
    }

    const x = clamp(Math.floor(normalizedX * this.canvasWidth), 0, this.canvasWidth - 1);
    const y = clamp(Math.floor(normalizedY * this.canvasHeight), 0, this.canvasHeight - 1);
    return this.objectIdBuffer[y * this.canvasWidth + x];
  }

  private clear(color: [number, number, number]) {
    for (let i = 0; i < this.colorData.length; i += 4) {
      this.colorData[i] = color[0];
      this.colorData[i + 1] = color[1];
      this.colorData[i + 2] = color[2];
      this.colorData[i + 3] = 255;
    }

    this.depthBuffer.fill(Number.POSITIVE_INFINITY);
    this.positionBuffer.fill(0);
    this.normalBuffer.fill(0);
    this.albedoBuffer.fill(0);
    this.gouraudBuffer.fill(0);
    this.edgeMask.fill(0);
    this.coverageMask.fill(0);
    this.objectIdBuffer.fill(-1);
  }

  private makeClipVertex(
    index: number,
    positionsView: Float32Array,
    positionsWorld: Float32Array,
    normalsWorld: Float32Array,
    uvs: Float32Array,
  ): ClipVertex {
    const base3 = index * 3;
    const base2 = index * 2;

    return {
      viewPosition: [
        positionsView[base3],
        positionsView[base3 + 1],
        positionsView[base3 + 2],
      ],
      worldPosition: [
        positionsWorld[base3],
        positionsWorld[base3 + 1],
        positionsWorld[base3 + 2],
      ],
      normal: [
        normalsWorld[base3],
        normalsWorld[base3 + 1],
        normalsWorld[base3 + 2],
      ],
      uv: [uvs[base2], uvs[base2 + 1]],
    };
  }

  private unwrapTriangleU(vertices: [ClipVertex, ClipVertex, ClipVertex]) {
    const adjusted = vertices.map((vertex) => ({
      viewPosition: [...vertex.viewPosition] as Vec3,
      worldPosition: [...vertex.worldPosition] as Vec3,
      normal: [...vertex.normal] as Vec3,
      uv: [...vertex.uv] as Vec2,
    })) as [ClipVertex, ClipVertex, ClipVertex];

    const values = adjusted.map((vertex) => vertex.uv[0]);
    const min = Math.min(...values);
    const max = Math.max(...values);

    if (max - min > 0.5) {
      for (const vertex of adjusted) {
        if (vertex.uv[0] < 0.5) {
          vertex.uv[0] += 1;
        }
      }
    }

    return adjusted;
  }

  private clipTriangle(vertices: ClipVertex[], near: number, far: number): ClipVertex[][] {
    let polygon = this.clipPolygonAgainstPlane(vertices, (vertex) => vertex.viewPosition[2] <= -near, -near);
    polygon = this.clipPolygonAgainstPlane(polygon, (vertex) => vertex.viewPosition[2] >= -far, -far);

    if (polygon.length < 3) {
      return [];
    }

    const output: ClipVertex[][] = [];

    for (let i = 1; i < polygon.length - 1; i += 1) {
      output.push([polygon[0], polygon[i], polygon[i + 1]]);
    }

    return output;
  }

  private clipPolygonAgainstPlane(
    polygon: ClipVertex[],
    inside: (vertex: ClipVertex) => boolean,
    planeZ: number,
  ) {
    if (polygon.length === 0) {
      return polygon;
    }

    const result: ClipVertex[] = [];
    let previous = polygon[polygon.length - 1];
    let previousInside = inside(previous);

    for (const current of polygon) {
      const currentInside = inside(current);

      if (currentInside !== previousInside) {
        result.push(this.interpolateClipVertex(previous, current, planeZ));
      }

      if (currentInside) {
        result.push(current);
      }

      previous = current;
      previousInside = currentInside;
    }

    return result;
  }

  private interpolateClipVertex(a: ClipVertex, b: ClipVertex, planeZ: number): ClipVertex {
    const t = (planeZ - a.viewPosition[2]) / (b.viewPosition[2] - a.viewPosition[2]);

    return {
      viewPosition: vec3.lerp(a.viewPosition, b.viewPosition, t),
      worldPosition: vec3.lerp(a.worldPosition, b.worldPosition, t),
      normal: vec3.normalize(vec3.lerp(a.normal, b.normal, t)),
      uv: vec2.lerp(a.uv, b.uv, t),
    };
  }

  private rasterizeTriangle(
    vertices: ClipVertex[],
    scene: RenderScene,
    objectId: number,
    shadowCamera: ShadowCamera | null,
  ) {
    const projected = vertices.map((vertex) => this.projectVertex(vertex, scene, shadowCamera)) as [
      ProjectedVertex,
      ProjectedVertex,
      ProjectedVertex,
    ];

    const x0 = projected[0].screenX;
    const y0 = projected[0].screenY;
    const x1 = projected[1].screenX;
    const y1 = projected[1].screenY;
    const x2 = projected[2].screenX;
    const y2 = projected[2].screenY;

    const area = edgeFunction(x0, y0, x1, y1, x2, y2);

    if (area === 0) {
      return;
    }

    const minX = clamp(Math.floor(Math.min(x0, x1, x2)), 0, this.canvasWidth - 1);
    const maxX = clamp(Math.ceil(Math.max(x0, x1, x2)), 0, this.canvasWidth - 1);
    const minY = clamp(Math.floor(Math.min(y0, y1, y2)), 0, this.canvasHeight - 1);
    const maxY = clamp(Math.ceil(Math.max(y0, y1, y2)), 0, this.canvasHeight - 1);

    const edgeLength0 = Math.hypot(x2 - x1, y2 - y1) || 1;
    const edgeLength1 = Math.hypot(x0 - x2, y0 - y2) || 1;
    const edgeLength2 = Math.hypot(x1 - x0, y1 - y0) || 1;

    for (let y = minY; y <= maxY; y += 1) {
      for (let x = minX; x <= maxX; x += 1) {
        const px = x + 0.5;
        const py = y + 0.5;

        const w0 = edgeFunction(x1, y1, x2, y2, px, py);
        const w1 = edgeFunction(x2, y2, x0, y0, px, py);
        const w2 = edgeFunction(x0, y0, x1, y1, px, py);

        if (!sameSign(w0, area) || !sameSign(w1, area) || !sameSign(w2, area)) {
          continue;
        }

        const lambda0 = w0 / area;
        const lambda1 = w1 / area;
        const lambda2 = w2 / area;
        const invW =
          lambda0 * projected[0].invW +
          lambda1 * projected[1].invW +
          lambda2 * projected[2].invW;

        const perspective0 = (lambda0 * projected[0].invW) / invW;
        const perspective1 = (lambda1 * projected[1].invW) / invW;
        const perspective2 = (lambda2 * projected[2].invW) / invW;
        const depth =
          perspective0 * projected[0].depth +
          perspective1 * projected[1].depth +
          perspective2 * projected[2].depth;

        const pixelIndex = y * this.canvasWidth + x;

        if (depth >= this.depthBuffer[pixelIndex]) {
          continue;
        }

        this.depthBuffer[pixelIndex] = depth;
        this.coverageMask[pixelIndex] = 1;
        this.objectIdBuffer[pixelIndex] = objectId;

        const normal = vec3.normalize([
          perspective0 * projected[0].normal[0] +
            perspective1 * projected[1].normal[0] +
            perspective2 * projected[2].normal[0],
          perspective0 * projected[0].normal[1] +
            perspective1 * projected[1].normal[1] +
            perspective2 * projected[2].normal[1],
          perspective0 * projected[0].normal[2] +
            perspective1 * projected[1].normal[2] +
            perspective2 * projected[2].normal[2],
        ]);

        const worldPosition: Vec3 = [
          perspective0 * projected[0].worldPosition[0] +
            perspective1 * projected[1].worldPosition[0] +
            perspective2 * projected[2].worldPosition[0],
          perspective0 * projected[0].worldPosition[1] +
            perspective1 * projected[1].worldPosition[1] +
            perspective2 * projected[2].worldPosition[1],
          perspective0 * projected[0].worldPosition[2] +
            perspective1 * projected[1].worldPosition[2] +
            perspective2 * projected[2].worldPosition[2],
        ];

        const uv: Vec2 = [
          perspective0 * projected[0].uv[0] +
            perspective1 * projected[1].uv[0] +
            perspective2 * projected[2].uv[0],
          perspective0 * projected[0].uv[1] +
            perspective1 * projected[1].uv[1] +
            perspective2 * projected[2].uv[1],
        ];

        const needsLighting = scene.shading === "gouraud" || scene.shading === "phong";
        const albedo: Vec3 = needsLighting ? this.sampleTexture(scene.texture, uv[0], uv[1]) : [1, 1, 1];
        const gouraud: Vec3 = [
          perspective0 * projected[0].gouraud[0] +
            perspective1 * projected[1].gouraud[0] +
            perspective2 * projected[2].gouraud[0],
          perspective0 * projected[0].gouraud[1] +
            perspective1 * projected[1].gouraud[1] +
            perspective2 * projected[2].gouraud[1],
          perspective0 * projected[0].gouraud[2] +
            perspective1 * projected[1].gouraud[2] +
            perspective2 * projected[2].gouraud[2],
        ];

        this.writeVec3(this.positionBuffer, pixelIndex, worldPosition);
        this.writeVec3(this.normalBuffer, pixelIndex, normal);
        this.writeVec3(this.albedoBuffer, pixelIndex, albedo);
        this.writeVec3(this.gouraudBuffer, pixelIndex, gouraud);

        const distance0 = Math.abs(w0) / edgeLength0;
        const distance1 = Math.abs(w1) / edgeLength1;
        const distance2 = Math.abs(w2) / edgeLength2;
        this.edgeMask[pixelIndex] = Math.min(distance0, distance1, distance2) < 0.95 ? 1 : 0;
      }
    }
  }

  private projectVertex(
    vertex: ClipVertex,
    scene: RenderScene,
    shadowCamera: ShadowCamera | null,
  ): ProjectedVertex {
    const clip = mat4.transformVec4(scene.projectionMatrix, [
      vertex.viewPosition[0],
      vertex.viewPosition[1],
      vertex.viewPosition[2],
      1,
    ]);

    const invW = 1 / clip[3];
    const ndcX = clip[0] * invW;
    const ndcY = clip[1] * invW;
    const ndcZ = clip[2] * invW;
    const needsLighting = scene.shading === "gouraud" || scene.shading === "phong";
    const albedo: Vec3 = needsLighting ? this.sampleTexture(scene.texture, vertex.uv[0], vertex.uv[1]) : [1, 1, 1];

    return {
      ...vertex,
      screenX: (ndcX * 0.5 + 0.5) * (this.canvasWidth - 1),
      screenY: (1 - (ndcY * 0.5 + 0.5)) * (this.canvasHeight - 1),
      invW,
      depth: ndcZ * 0.5 + 0.5,
      gouraud: needsLighting
        ? shadePoint(
            vertex.worldPosition,
            vertex.normal,
            scene.cameraPosition,
            scene.light,
            albedo,
            this.sampleShadow(vertex.worldPosition, vertex.normal, scene, shadowCamera),
          )
        : [0, 0, 0],
    };
  }

  private compose(scene: RenderScene, shadowCamera: ShadowCamera | null) {
    for (let pixelIndex = 0; pixelIndex < this.coverageMask.length; pixelIndex += 1) {
      if (this.coverageMask[pixelIndex] === 0) {
        continue;
      }

      let color: Vec3;

      switch (scene.shading) {
        case "normals":
          color = encodeNormal(this.readVec3(this.normalBuffer, pixelIndex));
          break;
        case "gouraud":
          color = scale255(this.readVec3(this.gouraudBuffer, pixelIndex));
          break;
        case "wireframe":
          color = this.edgeMask[pixelIndex] === 1 ? [19, 31, 44] : [246, 244, 237];
          break;
        case "phong":
          color = scale255(
            shadePoint(
              this.readVec3(this.positionBuffer, pixelIndex),
              this.readVec3(this.normalBuffer, pixelIndex),
              scene.cameraPosition,
              scene.light,
              this.readVec3(this.albedoBuffer, pixelIndex),
              this.sampleShadow(
                this.readVec3(this.positionBuffer, pixelIndex),
                this.readVec3(this.normalBuffer, pixelIndex),
                scene,
                shadowCamera,
              ),
            ),
          );
          break;
      }

      if (
        scene.selectedObjectId !== null &&
        this.objectIdBuffer[pixelIndex] === scene.selectedObjectId &&
        this.edgeMask[pixelIndex] === 1
      ) {
        color = [255, 220, 110];
      }

      const base = pixelIndex * 4;
      this.colorData[base] = color[0];
      this.colorData[base + 1] = color[1];
      this.colorData[base + 2] = color[2];
      this.colorData[base + 3] = 255;
    }
  }

  private createShadowCamera(scene: RenderScene): ShadowCamera {
    this.ensureShadowMap(scene.shadowMapSize);

    const radius = Math.max(1.5, scene.shadowRadius);
    const toTarget = vec3.sub(scene.shadowCenter, scene.light.position);
    const distance = Math.max(0.5, vec3.length(toTarget));
    const up: Vec3 =
      Math.abs(vec3.dot(vec3.normalize(toTarget), [0, 1, 0])) > 0.96 ? [0, 0, 1] : [0, 1, 0];
    const near = Math.max(0.1, distance - radius * 2.4);
    const far = distance + radius * 2.4;

    return {
      viewMatrix: mat4.lookAt(scene.light.position, scene.shadowCenter, up),
      projectionMatrix: mat4.orthographic(-radius, radius, -radius, radius, near, far),
      depthBuffer: this.shadowDepthBuffer,
      size: this.shadowMapSize,
    };
  }

  private rasterizeShadowMesh(
    mesh: IndexedMesh,
    positionsWorld: Float32Array,
    shadowCamera: ShadowCamera,
  ) {
    for (let triangle = 0; triangle < mesh.triangleCount; triangle += 1) {
      const base = triangle * 3;
      const projected = [
        this.projectShadowVertex(mesh.indices[base], positionsWorld, shadowCamera),
        this.projectShadowVertex(mesh.indices[base + 1], positionsWorld, shadowCamera),
        this.projectShadowVertex(mesh.indices[base + 2], positionsWorld, shadowCamera),
      ] as const;

      if (projected.every((vertex) => vertex.inside === false)) {
        continue;
      }

      const area = edgeFunction(
        projected[0].screenX,
        projected[0].screenY,
        projected[1].screenX,
        projected[1].screenY,
        projected[2].screenX,
        projected[2].screenY,
      );

      if (area === 0) {
        continue;
      }

      const minX = clamp(
        Math.floor(Math.min(projected[0].screenX, projected[1].screenX, projected[2].screenX)),
        0,
        shadowCamera.size - 1,
      );
      const maxX = clamp(
        Math.ceil(Math.max(projected[0].screenX, projected[1].screenX, projected[2].screenX)),
        0,
        shadowCamera.size - 1,
      );
      const minY = clamp(
        Math.floor(Math.min(projected[0].screenY, projected[1].screenY, projected[2].screenY)),
        0,
        shadowCamera.size - 1,
      );
      const maxY = clamp(
        Math.ceil(Math.max(projected[0].screenY, projected[1].screenY, projected[2].screenY)),
        0,
        shadowCamera.size - 1,
      );

      for (let y = minY; y <= maxY; y += 1) {
        for (let x = minX; x <= maxX; x += 1) {
          const px = x + 0.5;
          const py = y + 0.5;
          const w0 = edgeFunction(
            projected[1].screenX,
            projected[1].screenY,
            projected[2].screenX,
            projected[2].screenY,
            px,
            py,
          );
          const w1 = edgeFunction(
            projected[2].screenX,
            projected[2].screenY,
            projected[0].screenX,
            projected[0].screenY,
            px,
            py,
          );
          const w2 = edgeFunction(
            projected[0].screenX,
            projected[0].screenY,
            projected[1].screenX,
            projected[1].screenY,
            px,
            py,
          );

          if (!sameSign(w0, area) || !sameSign(w1, area) || !sameSign(w2, area)) {
            continue;
          }

          const lambda0 = w0 / area;
          const lambda1 = w1 / area;
          const lambda2 = w2 / area;
          const depth =
            lambda0 * projected[0].depth +
            lambda1 * projected[1].depth +
            lambda2 * projected[2].depth;

          const pixelIndex = y * shadowCamera.size + x;

          if (depth < shadowCamera.depthBuffer[pixelIndex]) {
            shadowCamera.depthBuffer[pixelIndex] = depth;
          }
        }
      }
    }
  }

  private projectShadowVertex(index: number, positionsWorld: Float32Array, shadowCamera: ShadowCamera) {
    const base = index * 3;
    const worldPosition: Vec3 = [
      positionsWorld[base],
      positionsWorld[base + 1],
      positionsWorld[base + 2],
    ];
    const lightView = mat4.transformPoint(shadowCamera.viewMatrix, worldPosition);
    const clip = mat4.transformVec4(shadowCamera.projectionMatrix, [
      lightView[0],
      lightView[1],
      lightView[2],
      1,
    ]);
    const invW = clip[3] === 0 ? 1 : 1 / clip[3];
    const ndcX = clip[0] * invW;
    const ndcY = clip[1] * invW;
    const ndcZ = clip[2] * invW;

    return {
      screenX: (ndcX * 0.5 + 0.5) * (shadowCamera.size - 1),
      screenY: (1 - (ndcY * 0.5 + 0.5)) * (shadowCamera.size - 1),
      depth: ndcZ * 0.5 + 0.5,
      inside: ndcX >= -1 && ndcX <= 1 && ndcY >= -1 && ndcY <= 1 && ndcZ >= -1 && ndcZ <= 1,
    };
  }

  private sampleShadow(
    worldPosition: Vec3,
    normal: Vec3,
    scene: RenderScene,
    shadowCamera: ShadowCamera | null,
  ) {
    if (!shadowCamera || !scene.light.shadowEnabled) {
      return 1;
    }

    const projected = this.projectShadowVertexFromWorld(worldPosition, shadowCamera);

    if (!projected.inside) {
      return 1;
    }

    const lightDirection = vec3.normalize(vec3.sub(scene.light.position, worldPosition));
    const slope = 1 - Math.max(0, vec3.dot(vec3.normalize(normal), lightDirection));
    const bias = scene.light.shadowBias * Math.max(1, slope * 2.2);
    let occluded = 0;
    let total = 0;

    for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
      for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
        const sampleX = clamp(Math.round(projected.screenX) + offsetX, 0, shadowCamera.size - 1);
        const sampleY = clamp(Math.round(projected.screenY) + offsetY, 0, shadowCamera.size - 1);
        const sampleDepth = shadowCamera.depthBuffer[sampleY * shadowCamera.size + sampleX];
        total += 1;

        if (projected.depth - bias > sampleDepth) {
          occluded += 1;
        }
      }
    }

    if (total === 0) {
      return 1;
    }

    return 1 - (occluded / total) * scene.light.shadowStrength;
  }

  private projectShadowVertexFromWorld(worldPosition: Vec3, shadowCamera: ShadowCamera) {
    const lightView = mat4.transformPoint(shadowCamera.viewMatrix, worldPosition);
    const clip = mat4.transformVec4(shadowCamera.projectionMatrix, [
      lightView[0],
      lightView[1],
      lightView[2],
      1,
    ]);
    const invW = clip[3] === 0 ? 1 : 1 / clip[3];
    const ndcX = clip[0] * invW;
    const ndcY = clip[1] * invW;
    const ndcZ = clip[2] * invW;

    return {
      screenX: (ndcX * 0.5 + 0.5) * (shadowCamera.size - 1),
      screenY: (1 - (ndcY * 0.5 + 0.5)) * (shadowCamera.size - 1),
      depth: ndcZ * 0.5 + 0.5,
      inside: ndcX >= -1 && ndcX <= 1 && ndcY >= -1 && ndcY <= 1 && ndcZ >= -1 && ndcZ <= 1,
    };
  }

  private sampleTexture(texture: TextureData, rawU: number, rawV: number): Vec3 {
    const u = rawU - Math.floor(rawU);
    const v = clamp(rawV, 0, 1);
    const x = Math.min(texture.width - 1, Math.floor(u * (texture.width - 1)));
    const y = Math.min(texture.height - 1, Math.floor((1 - v) * (texture.height - 1)));
    const index = (y * texture.width + x) * 4;

    return [
      texture.pixels[index] / 255,
      texture.pixels[index + 1] / 255,
      texture.pixels[index + 2] / 255,
    ];
  }

  private writeVec3(buffer: Float32Array, pixelIndex: number, value: Vec3) {
    const base = pixelIndex * 3;
    buffer[base] = value[0];
    buffer[base + 1] = value[1];
    buffer[base + 2] = value[2];
  }

  private readVec3(buffer: Float32Array, pixelIndex: number): Vec3 {
    const base = pixelIndex * 3;
    return [buffer[base], buffer[base + 1], buffer[base + 2]];
  }
}

export async function loadTextureData(url: string): Promise<TextureData> {
  const image = new Image();
  image.src = url;
  await image.decode();

  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;

  const context = canvas.getContext("2d");

  if (!context) {
    throw new Error("Could not create temporary 2D context");
  }

  context.drawImage(image, 0, 0);
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);

  return {
    width: canvas.width,
    height: canvas.height,
    pixels: imageData.data,
  };
}

function edgeFunction(ax: number, ay: number, bx: number, by: number, px: number, py: number) {
  return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

function isBackFacing(a: Vec3, b: Vec3, c: Vec3) {
  const ab = vec3.sub(b, a);
  const ac = vec3.sub(c, a);
  const faceNormal = vec3.cross(ab, ac);
  const center: Vec3 = [
    (a[0] + b[0] + c[0]) / 3,
    (a[1] + b[1] + c[1]) / 3,
    (a[2] + b[2] + c[2]) / 3,
  ];
  return vec3.dot(faceNormal, vec3.scale(center, -1)) <= 0;
}

function sameSign(value: number, area: number) {
  return area < 0 ? value <= 0 : value >= 0;
}

function scale255(color: Vec3): Vec3 {
  return [
    Math.round(clamp(color[0], 0, 1) * 255),
    Math.round(clamp(color[1], 0, 1) * 255),
    Math.round(clamp(color[2], 0, 1) * 255),
  ];
}

function encodeNormal(normal: Vec3): Vec3 {
  return [
    Math.round((normal[0] * 0.5 + 0.5) * 255),
    Math.round((normal[1] * 0.5 + 0.5) * 255),
    Math.round((normal[2] * 0.5 + 0.5) * 255),
  ];
}

function shadePoint(
  position: Vec3,
  normal: Vec3,
  cameraPosition: Vec3,
  light: LightSettings,
  albedo: Vec3,
  shadowFactor: number,
): Vec3 {
  const n = vec3.normalize(normal);
  const lightDirection = vec3.normalize(vec3.sub(light.position, position));
  const viewDirection = vec3.normalize(vec3.sub(cameraPosition, position));
  const halfVector = vec3.normalize(vec3.add(lightDirection, viewDirection));
  const lambert = Math.max(0, vec3.dot(n, lightDirection));

  const ambient = light.ambient;
  const diffuse = lambert * light.diffuse * shadowFactor;
  const specular =
    (lambert > 0 ? Math.pow(Math.max(0, vec3.dot(n, halfVector)), light.specularPower) : 0) *
    light.specular *
    shadowFactor;

  return [
    albedo[0] * (ambient + diffuse) + specular,
    albedo[1] * (ambient + diffuse) + specular,
    albedo[2] * (ambient + diffuse) + specular,
  ];
}
