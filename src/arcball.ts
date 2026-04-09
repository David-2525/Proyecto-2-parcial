import { clamp, quat, type Quat, type Vec3 } from "./math";

export class ArcballController {
  rotation: Quat = quat.identity();
  zoom = 1;

  private readonly canvas: HTMLCanvasElement;
  private dragging = false;
  private dragStartRotation: Quat = quat.identity();
  private dragStartVector: Vec3 = [0, 0, 1];

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  reset() {
    this.rotation = quat.identity();
    this.zoom = 1;
  }

  setZoom(value: number) {
    this.zoom = clamp(value, 0.45, 3.2);
  }

  beginDrag(clientX: number, clientY: number) {
    this.dragging = true;
    this.dragStartVector = this.projectToSphere(clientX, clientY);
    this.dragStartRotation = this.rotation;
  }

  dragTo(clientX: number, clientY: number) {
    if (!this.dragging) {
      return;
    }

    const currentVector = this.projectToSphere(clientX, clientY);
    const delta = quat.fromUnitVectors(this.dragStartVector, currentVector);
    this.rotation = quat.normalize(quat.multiply(delta, this.dragStartRotation));
  }

  endDrag() {
    this.dragging = false;
  }

  applyWheel(deltaY: number) {
    const factor = Math.exp(-deltaY * 0.0015);
    this.zoom = clamp(this.zoom * factor, 0.45, 3.2);
  }

  nudge(deltaX: number, deltaY: number) {
    const yaw = quat.fromAxisAngle([0, 1, 0], deltaX * 0.004);
    const pitch = quat.fromAxisAngle([1, 0, 0], deltaY * 0.004);
    this.rotation = quat.normalize(quat.multiply(yaw, quat.multiply(pitch, this.rotation)));
  }

  private projectToSphere(clientX: number, clientY: number): Vec3 {
    const rect = this.canvas.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * 2 - 1;
    const y = 1 - ((clientY - rect.top) / rect.height) * 2;
    const lengthSquared = x * x + y * y;

    if (lengthSquared <= 1) {
      return [x, y, Math.sqrt(1 - lengthSquared)];
    }

    const scale = 1 / Math.sqrt(lengthSquared);
    return [x * scale, y * scale, 0];
  }
}
