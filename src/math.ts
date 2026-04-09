export type Vec2 = [number, number];
export type Vec3 = [number, number, number];
export type Vec4 = [number, number, number, number];
export type Quat = [number, number, number, number];
export type Mat4 = Float32Array;

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export const vec2 = {
  lerp(a: Vec2, b: Vec2, t: number): Vec2 {
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  },
};

export const vec3 = {
  add(a: Vec3, b: Vec3): Vec3 {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  },
  sub(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  },
  scale(v: Vec3, scalar: number): Vec3 {
    return [v[0] * scalar, v[1] * scalar, v[2] * scalar];
  },
  dot(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  },
  cross(a: Vec3, b: Vec3): Vec3 {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  },
  length(v: Vec3): number {
    return Math.hypot(v[0], v[1], v[2]);
  },
  normalize(v: Vec3): Vec3 {
    const length = vec3.length(v);

    if (length === 0) {
      return [0, 0, 0];
    }

    return [v[0] / length, v[1] / length, v[2] / length];
  },
  lerp(a: Vec3, b: Vec3, t: number): Vec3 {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
    ];
  },
};

export const quat = {
  identity(): Quat {
    return [0, 0, 0, 1];
  },
  conjugate(q: Quat): Quat {
    return [-q[0], -q[1], -q[2], q[3]];
  },
  fromAxisAngle(axis: Vec3, angle: number): Quat {
    const normalized = vec3.normalize(axis);
    const halfAngle = angle * 0.5;
    const sinHalf = Math.sin(halfAngle);

    return [
      normalized[0] * sinHalf,
      normalized[1] * sinHalf,
      normalized[2] * sinHalf,
      Math.cos(halfAngle),
    ];
  },
  normalize(q: Quat): Quat {
    const length = Math.hypot(q[0], q[1], q[2], q[3]);

    if (length === 0) {
      return [0, 0, 0, 1];
    }

    return [q[0] / length, q[1] / length, q[2] / length, q[3] / length];
  },
  multiply(a: Quat, b: Quat): Quat {
    return [
      a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
      a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
      a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
      a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ];
  },
  fromUnitVectors(from: Vec3, to: Vec3): Quat {
    const dot = vec3.dot(from, to);

    if (dot < -0.999999) {
      const axis =
        Math.abs(from[0]) > Math.abs(from[2])
          ? vec3.normalize([-from[1], from[0], 0])
          : vec3.normalize([0, -from[2], from[1]]);

      return [axis[0], axis[1], axis[2], 0];
    }

    const cross = vec3.cross(from, to);
    return quat.normalize([cross[0], cross[1], cross[2], 1 + dot]);
  },
};

export const mat4 = {
  identity(): Mat4 {
    const matrix = new Float32Array(16);
    matrix[0] = 1;
    matrix[5] = 1;
    matrix[10] = 1;
    matrix[15] = 1;
    return matrix;
  },
  multiply(a: Mat4, b: Mat4): Mat4 {
    const out = new Float32Array(16);

    for (let column = 0; column < 4; column += 1) {
      for (let row = 0; row < 4; row += 1) {
        out[column * 4 + row] =
          a[0 * 4 + row] * b[column * 4 + 0] +
          a[1 * 4 + row] * b[column * 4 + 1] +
          a[2 * 4 + row] * b[column * 4 + 2] +
          a[3 * 4 + row] * b[column * 4 + 3];
      }
    }

    return out;
  },
  translation(tx: number, ty: number, tz: number): Mat4 {
    const matrix = mat4.identity();
    matrix[12] = tx;
    matrix[13] = ty;
    matrix[14] = tz;
    return matrix;
  },
  scaling(sx: number, sy: number, sz: number): Mat4 {
    const matrix = mat4.identity();
    matrix[0] = sx;
    matrix[5] = sy;
    matrix[10] = sz;
    return matrix;
  },
  fromQuat(q: Quat): Mat4 {
    const [x, y, z, w] = quat.normalize(q);
    const xx = x * x;
    const yy = y * y;
    const zz = z * z;
    const xy = x * y;
    const xz = x * z;
    const yz = y * z;
    const wx = w * x;
    const wy = w * y;
    const wz = w * z;

    const matrix = mat4.identity();
    matrix[0] = 1 - 2 * (yy + zz);
    matrix[1] = 2 * (xy + wz);
    matrix[2] = 2 * (xz - wy);

    matrix[4] = 2 * (xy - wz);
    matrix[5] = 1 - 2 * (xx + zz);
    matrix[6] = 2 * (yz + wx);

    matrix[8] = 2 * (xz + wy);
    matrix[9] = 2 * (yz - wx);
    matrix[10] = 1 - 2 * (xx + yy);

    return matrix;
  },
  perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
    const f = 1 / Math.tan(fovY * 0.5);
    const matrix = new Float32Array(16);
    matrix[0] = f / aspect;
    matrix[5] = f;
    matrix[10] = (far + near) / (near - far);
    matrix[11] = -1;
    matrix[14] = (2 * far * near) / (near - far);
    return matrix;
  },
  orthographic(left: number, right: number, bottom: number, top: number, near: number, far: number): Mat4 {
    const matrix = new Float32Array(16);
    matrix[0] = 2 / (right - left);
    matrix[5] = 2 / (top - bottom);
    matrix[10] = -2 / (far - near);
    matrix[12] = -(right + left) / (right - left);
    matrix[13] = -(top + bottom) / (top - bottom);
    matrix[14] = -(far + near) / (far - near);
    matrix[15] = 1;
    return matrix;
  },
  lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
    const z = vec3.normalize(vec3.sub(eye, target));
    const x = vec3.normalize(vec3.cross(up, z));
    const y = vec3.cross(z, x);

    const matrix = mat4.identity();
    matrix[0] = x[0];
    matrix[1] = y[0];
    matrix[2] = z[0];

    matrix[4] = x[1];
    matrix[5] = y[1];
    matrix[6] = z[1];

    matrix[8] = x[2];
    matrix[9] = y[2];
    matrix[10] = z[2];

    matrix[12] = -vec3.dot(x, eye);
    matrix[13] = -vec3.dot(y, eye);
    matrix[14] = -vec3.dot(z, eye);

    return matrix;
  },
  transformVec4(matrix: Mat4, vector: Vec4): Vec4 {
    return [
      matrix[0] * vector[0] +
        matrix[4] * vector[1] +
        matrix[8] * vector[2] +
        matrix[12] * vector[3],
      matrix[1] * vector[0] +
        matrix[5] * vector[1] +
        matrix[9] * vector[2] +
        matrix[13] * vector[3],
      matrix[2] * vector[0] +
        matrix[6] * vector[1] +
        matrix[10] * vector[2] +
        matrix[14] * vector[3],
      matrix[3] * vector[0] +
        matrix[7] * vector[1] +
        matrix[11] * vector[2] +
        matrix[15] * vector[3],
    ];
  },
  transformPoint(matrix: Mat4, vector: Vec3): Vec3 {
    const out = mat4.transformVec4(matrix, [vector[0], vector[1], vector[2], 1]);
    return [out[0], out[1], out[2]];
  },
  transformVector(matrix: Mat4, vector: Vec3): Vec3 {
    const out = mat4.transformVec4(matrix, [vector[0], vector[1], vector[2], 0]);
    return [out[0], out[1], out[2]];
  },
};
