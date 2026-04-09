import { vec3, type Vec2, type Vec3 } from "./math";
import { computeMeshBounds, type IndexedMesh } from "./mesh";

interface ObjIndex {
  position: number;
  texcoord: number | null;
  normal: number | null;
}

export async function loadOBJ(url: string, name = url): Promise<IndexedMesh> {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Could not load OBJ: ${url}`);
  }

  return parseOBJ(await response.text(), name);
}

export function parseOBJ(text: string, name = "mesh"): IndexedMesh {
  const rawPositions: Vec3[] = [];
  const rawTexcoords: Vec2[] = [];
  const rawNormals: Vec3[] = [];

  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  const vertexMap = new Map<string, number>();

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();

    if (!line || line.startsWith("#")) {
      continue;
    }

    const parts = line.split(/\s+/);

    switch (parts[0]) {
      case "v":
        rawPositions.push([
          Number(parts[1]),
          Number(parts[2]),
          Number(parts[3]),
        ]);
        break;
      case "vt":
        rawTexcoords.push([Number(parts[1]), Number(parts[2])]);
        break;
      case "vn":
        rawNormals.push([
          Number(parts[1]),
          Number(parts[2]),
          Number(parts[3]),
        ]);
        break;
      case "f": {
        const face = parts
          .slice(1)
          .map((token) =>
            parseFaceToken(token, rawPositions.length, rawTexcoords.length, rawNormals.length),
          );

        for (let i = 1; i < face.length - 1; i += 1) {
          indices.push(
            getVertexIndex(face[0], rawPositions, rawTexcoords, rawNormals, vertexMap, positions, uvs, normals),
            getVertexIndex(face[i], rawPositions, rawTexcoords, rawNormals, vertexMap, positions, uvs, normals),
            getVertexIndex(face[i + 1], rawPositions, rawTexcoords, rawNormals, vertexMap, positions, uvs, normals),
          );
        }
        break;
      }
      default:
        break;
    }
  }

  const positionData = Float32Array.from(positions);
  const normalData = Float32Array.from(normals);
  const uvData = Float32Array.from(uvs);
  const indexData = Uint32Array.from(indices);

  const mesh: IndexedMesh = {
    name,
    positions: positionData,
    normals: normalData,
    uvs: uvData,
    indices: indexData,
    faceNormals: new Float32Array((indexData.length / 3) * 3),
    vertexCount: positionData.length / 3,
    triangleCount: indexData.length / 3,
    ...computeMeshBounds(positionData),
  };

  recomputeNormals(mesh);
  return mesh;
}

function parseFaceToken(
  token: string,
  vertexCount: number,
  texcoordCount: number,
  normalCount: number,
): ObjIndex {
  const [position, texcoord, normal] = token.split("/");

  return {
    position: resolveIndex(position, vertexCount),
    texcoord: texcoord ? resolveIndex(texcoord, texcoordCount) : null,
    normal: normal ? resolveIndex(normal, normalCount) : null,
  };
}

function resolveIndex(value: string, count: number): number {
  const parsed = Number(value);
  return parsed > 0 ? parsed - 1 : count + parsed;
}

function getVertexIndex(
  index: ObjIndex,
  rawPositions: Vec3[],
  rawTexcoords: Vec2[],
  rawNormals: Vec3[],
  vertexMap: Map<string, number>,
  positions: number[],
  uvs: number[],
  normals: number[],
): number {
  const key = `${index.position}/${index.texcoord ?? ""}/${index.normal ?? ""}`;
  const cached = vertexMap.get(key);

  if (cached !== undefined) {
    return cached;
  }

  const position = rawPositions[index.position];
  const texcoord = index.texcoord !== null ? rawTexcoords[index.texcoord] : [0, 0];
  const normal = index.normal !== null ? rawNormals[index.normal] : [0, 0, 0];
  const vertexIndex = positions.length / 3;

  positions.push(position[0], position[1], position[2]);
  uvs.push(texcoord[0], texcoord[1]);
  normals.push(normal[0], normal[1], normal[2]);

  vertexMap.set(key, vertexIndex);
  return vertexIndex;
}

export function recomputeNormals(mesh: IndexedMesh) {
  mesh.normals.fill(0);

  for (let triangle = 0; triangle < mesh.triangleCount; triangle += 1) {
    const base = triangle * 3;
    const ia = mesh.indices[base] * 3;
    const ib = mesh.indices[base + 1] * 3;
    const ic = mesh.indices[base + 2] * 3;

    const a: Vec3 = [mesh.positions[ia], mesh.positions[ia + 1], mesh.positions[ia + 2]];
    const b: Vec3 = [mesh.positions[ib], mesh.positions[ib + 1], mesh.positions[ib + 2]];
    const c: Vec3 = [mesh.positions[ic], mesh.positions[ic + 1], mesh.positions[ic + 2]];

    const faceNormal = vec3.normalize(vec3.cross(vec3.sub(b, a), vec3.sub(c, a)));

    mesh.faceNormals[base] = faceNormal[0];
    mesh.faceNormals[base + 1] = faceNormal[1];
    mesh.faceNormals[base + 2] = faceNormal[2];

    addNormal(mesh.normals, ia, faceNormal);
    addNormal(mesh.normals, ib, faceNormal);
    addNormal(mesh.normals, ic, faceNormal);
  }

  for (let vertex = 0; vertex < mesh.vertexCount; vertex += 1) {
    const base = vertex * 3;
    const normal = vec3.normalize([
      mesh.normals[base],
      mesh.normals[base + 1],
      mesh.normals[base + 2],
    ]);

    if (vec3.length(normal) === 0) {
      mesh.normals[base] = 0;
      mesh.normals[base + 1] = 1;
      mesh.normals[base + 2] = 0;
      continue;
    }

    mesh.normals[base] = normal[0];
    mesh.normals[base + 1] = normal[1];
    mesh.normals[base + 2] = normal[2];
  }
}

function addNormal(buffer: Float32Array, index: number, normal: Vec3) {
  buffer[index] += normal[0];
  buffer[index + 1] += normal[1];
  buffer[index + 2] += normal[2];
}
