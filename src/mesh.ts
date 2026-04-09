import { mat4, vec3, type Mat4, type Vec3 } from "./math";

export interface IndexedMesh {
  name: string;
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  indices: Uint32Array;
  faceNormals: Float32Array;
  vertexCount: number;
  triangleCount: number;
  min: Vec3;
  max: Vec3;
  center: Vec3;
  radius: number;
}

export type BoundsDefinition =
  | {
      kind: "sphere";
      center: Vec3;
      radius: number;
    }
  | {
      kind: "box";
      center: Vec3;
      min: Vec3;
      max: Vec3;
    };

export interface ModelPreset {
  id: string;
  label: string;
  source: "obj" | "cube" | "sphere";
  url?: string;
  bounds: BoundsDefinition;
}

export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: "teapot",
    label: "Utah Teapot",
    source: "obj",
    url: `${import.meta.env.BASE_URL}models/teapot.obj`,
    bounds: {
      kind: "box",
      center: [0.217, 1.575, 0],
      min: [-3, 0, -2],
      max: [3.434, 3.15, 2],
    },
  },
  {
    id: "beacon",
    label: "KAUST Beacon",
    source: "obj",
    url: `${import.meta.env.BASE_URL}models/KAUST_Beacon.obj`,
    bounds: {
      kind: "sphere",
      center: [125, 125, 125],
      radius: 125,
    },
  },
  {
    id: "cube",
    label: "Cube",
    source: "cube",
    bounds: {
      kind: "sphere",
      center: [0, 0, 0],
      radius: 1,
    },
  },
  {
    id: "sphere",
    label: "Sphere",
    source: "sphere",
    bounds: {
      kind: "sphere",
      center: [0, 0, 0],
      radius: 1,
    },
  },
];

export function computeMeshBounds(positions: Float32Array) {
  const min: Vec3 = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const max: Vec3 = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];

  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];

    if (x < min[0]) min[0] = x;
    if (y < min[1]) min[1] = y;
    if (z < min[2]) min[2] = z;

    if (x > max[0]) max[0] = x;
    if (y > max[1]) max[1] = y;
    if (z > max[2]) max[2] = z;
  }

  const center: Vec3 = [
    (min[0] + max[0]) * 0.5,
    (min[1] + max[1]) * 0.5,
    (min[2] + max[2]) * 0.5,
  ];

  let radius = 0;

  for (let i = 0; i < positions.length; i += 3) {
    const point: Vec3 = [positions[i], positions[i + 1], positions[i + 2]];
    radius = Math.max(radius, vec3.length(vec3.sub(point, center)));
  }

  return { min, max, center, radius };
}

export function getBoundsSphere(bounds: BoundsDefinition) {
  if (bounds.kind === "sphere") {
    return bounds;
  }

  return {
    center: bounds.center,
    radius: vec3.length(vec3.sub(bounds.max, bounds.center)),
  };
}

export function createFitTransform(bounds: BoundsDefinition): {
  center: Vec3;
  radius: number;
  matrix: Mat4;
} {
  const sphere = getBoundsSphere(bounds);
  const translate = mat4.translation(-sphere.center[0], -sphere.center[1], -sphere.center[2]);
  const scale = mat4.scaling(1 / sphere.radius, 1 / sphere.radius, 1 / sphere.radius);

  return {
    center: sphere.center,
    radius: sphere.radius,
    matrix: mat4.multiply(scale, translate),
  };
}

export function generateSphericalUVs(mesh: IndexedMesh, center: Vec3) {
  for (let i = 0; i < mesh.vertexCount; i += 1) {
    const positionBase = i * 3;
    const uvBase = i * 2;
    const direction = vec3.normalize([
      mesh.positions[positionBase] - center[0],
      mesh.positions[positionBase + 1] - center[1],
      mesh.positions[positionBase + 2] - center[2],
    ]);

    mesh.uvs[uvBase] = 0.5 + Math.atan2(direction[2], direction[0]) / (Math.PI * 2);
    mesh.uvs[uvBase + 1] = 0.5 - Math.asin(direction[1]) / Math.PI;
  }
}
