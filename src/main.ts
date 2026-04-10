import "./style.css";
import { ArcballController } from "./arcball";
import { mat4, vec3, type Vec3 } from "./math";
import { loadOBJ, parseOBJ, recomputeNormals } from "./obj-loader";
import {
  MODEL_PRESETS,
  computeMeshBounds,
  createFitTransform,
  generateSphericalUVs,
  type IndexedMesh,
  type ModelPreset,
} from "./mesh";
import {
  SoftwareRenderer,
  loadTextureData,
  type LightSettings,
  type RenderObject,
  type RenderStats,
  type ShadingMode,
  type TextureData,
} from "./software-renderer";

interface LoadedAsset {
  preset: ModelPreset;
  mesh: IndexedMesh;
  fit: ReturnType<typeof createFitTransform>;
}

interface SceneObject {
  id: number;
  assetId: string;
  label: string;
  position: Vec3;
  scale: number;
}

type DragMode = "none" | "orbit" | "move";

interface TexturePreset {
  id: string;
  label: string;
  key: string;
  url: string;
}

interface LightControlDefinition {
  id: string;
  label: string;
  min: number;
  max: number;
  step: number;
  getValue: () => number;
  setValue: (value: number) => void;
}

const canvas = document.querySelector("#gfx-main") as HTMLCanvasElement | null;

if (!canvas) {
  throw new Error("Canvas not found");
}

const appCanvas = canvas;
const hud = document.createElement("div");
hud.className = "hud";
document.body.append(hud);

const toolbar = document.createElement("div");
toolbar.className = "toolbar";
document.body.append(toolbar);

const lightPanel = document.createElement("div");
lightPanel.className = "light-panel";
document.body.append(lightPanel);

const fileInput = document.createElement("input");
fileInput.type = "file";
fileInput.accept = ".obj";
fileInput.hidden = true;
document.body.append(fileInput);

const renderer = new SoftwareRenderer(appCanvas);
const arcball = new ArcballController(appCanvas);
const assetCache = new Map<string, LoadedAsset>();
const textureCache = new Map<string, TextureData>();
const sceneObjects: SceneObject[] = [];

const TEXTURE_PRESETS: TexturePreset[] = [
  { id: "tex1", label: "Texture 1", key: "H", url: `${import.meta.env.BASE_URL}textures/tex1.png` },
  { id: "tex2", label: "Texture 2", key: "J", url: `${import.meta.env.BASE_URL}textures/tex2.jpg` },
  { id: "tex3", label: "Texture 3", key: "K", url: `${import.meta.env.BASE_URL}textures/tex3.jpg` },
  { id: "tex4", label: "Texture 4", key: "L", url: `${import.meta.env.BASE_URL}textures/tex4.jpg` },
];

const DEFAULT_LIGHT: LightSettings = {
  position: [3.0, 4.0, 3.0],
  ambient: 0.25,
  diffuse: 1.0,
  specular: 0.6,
  specularPower: 48,
  shadowEnabled: true,
  shadowStrength: 0.5,
  shadowBias: 0.003,
};

const state = {
  shading: "phong" as ShadingMode,
  selectedId: null as number | null,
  texture: null as TextureData | null,
  textureId: "tex4",
  loadingMessage: "Loading texture...",
  interactive: false,
  cameraTarget: [0, 0, 0] as Vec3,
  light: cloneLight(DEFAULT_LIGHT),
};

const LIGHT_CONTROLS: LightControlDefinition[] = [
  {
    id: "position-x",
    label: "Light X",
    min: -8,
    max: 8,
    step: 0.1,
    getValue: () => state.light.position[0],
    setValue: (value) => {
      state.light.position[0] = value;
    },
  },
  {
    id: "position-y",
    label: "Light Y",
    min: -1,
    max: 8,
    step: 0.1,
    getValue: () => state.light.position[1],
    setValue: (value) => {
      state.light.position[1] = value;
    },
  },
  {
    id: "position-z",
    label: "Light Z",
    min: -8,
    max: 8,
    step: 0.1,
    getValue: () => state.light.position[2],
    setValue: (value) => {
      state.light.position[2] = value;
    },
  },
  {
    id: "ambient",
    label: "Ambient",
    min: 0,
    max: 0.8,
    step: 0.01,
    getValue: () => state.light.ambient,
    setValue: (value) => {
      state.light.ambient = value;
    },
  },
  {
    id: "diffuse",
    label: "Diffuse",
    min: 0,
    max: 1.6,
    step: 0.01,
    getValue: () => state.light.diffuse,
    setValue: (value) => {
      state.light.diffuse = value;
    },
  },
  {
    id: "specular",
    label: "Specular",
    min: 0,
    max: 1.4,
    step: 0.01,
    getValue: () => state.light.specular,
    setValue: (value) => {
      state.light.specular = value;
    },
  },
  {
    id: "shininess",
    label: "Shininess",
    min: 4,
    max: 96,
    step: 1,
    getValue: () => state.light.specularPower,
    setValue: (value) => {
      state.light.specularPower = value;
    },
  },
  {
    id: "shadow-strength",
    label: "Shadow",
    min: 0,
    max: 1,
    step: 0.01,
    getValue: () => state.light.shadowStrength,
    setValue: (value) => {
      state.light.shadowStrength = value;
    },
  },
  {
    id: "shadow-bias",
    label: "Bias",
    min: 0.0005,
    max: 0.03,
    step: 0.0005,
    getValue: () => state.light.shadowBias,
    setValue: (value) => {
      state.light.shadowBias = value;
    },
  },
];

const dragState = {
  mode: "none" as DragMode,
  startClientX: 0,
  startClientY: 0,
  lastClientX: 0,
  lastClientY: 0,
  moved: false,
  pointerDownPickId: -1,
  planePoint: [0, 0, 0] as Vec3,
  planeNormal: [0, 0, 1] as Vec3,
  positionOffset: [0, 0, 0] as Vec3,
};

let pendingRender = false;
let interactiveTimer = 0;
let nextObjectId = 1;
let nextUploadedAssetId = 1;

appCanvas.addEventListener("pointerdown", handlePointerDown);
window.addEventListener("pointermove", handlePointerMove);
window.addEventListener("pointerup", handlePointerUp);
window.addEventListener("pointercancel", handlePointerCancel);

appCanvas.addEventListener("dragover", (event) => {
  event.preventDefault();

  if (event.dataTransfer) {
    event.dataTransfer.dropEffect = "copy";
  }
});

appCanvas.addEventListener("drop", (event) => {
  event.preventDefault();
  const file = getFirstUploadedFile(event.dataTransfer?.files ?? null);

  if (!file) {
    window.alert("Drop an OBJ file to import a model.");
    return;
  }

  void importUploadedObject(file);
});

appCanvas.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    beginInteractiveRender();

    if (event.ctrlKey || event.metaKey || event.altKey) {
      arcball.applyWheel(event.deltaY);
    } else {
      arcball.nudge(-event.deltaX, event.deltaY);
    }

    scheduleRender();
    updateHud();
    endInteractiveRenderSoon();
  },
  { passive: false },
);

toolbar.addEventListener("click", (event) => {
  const target = event.target as HTMLElement | null;
  const button = target?.closest("button[data-action]") as HTMLButtonElement | null;

  if (!button) {
    return;
  }

  void handleToolbarAction(button.dataset.action ?? "");
});

lightPanel.addEventListener("input", handleLightPanelInput);
lightPanel.addEventListener("change", handleLightPanelChange);
lightPanel.addEventListener("click", handleLightPanelClick);

fileInput.addEventListener("change", () => {
  const file = getFirstUploadedFile(fileInput.files);
  fileInput.value = "";

  if (!file) {
    return;
  }

  void importUploadedObject(file);
});

window.addEventListener("resize", () => {
  renderer.resize();
  scheduleRender();
});

window.addEventListener("keydown", async (event) => {
  const key = event.key.toLowerCase();

  if (key === "t" || key === "1") {
    await addObject("teapot");
    return;
  }

  if (key === "b" || key === "2") {
    await addObject("beacon");
    return;
  }

  if (key === "c") {
    await addObject("cube");
    return;
  }

  if (key === "s" || key === "o") {
    await addObject("sphere");
    return;
  }

  if (key === "u") {
    event.preventDefault();
    openModelFilePicker();
    return;
  }

  if (key === "h") {
    await setTexture("tex1");
    return;
  }

  if (key === "j") {
    await setTexture("tex2");
    return;
  }

  if (key === "k") {
    await setTexture("tex3");
    return;
  }

  if (key === "l") {
    await setTexture("tex4");
    return;
  }

  if (key === "=" || key === "+") {
    beginInteractiveRender();
    arcball.applyWheel(-140);
    scheduleRender();
    updateHud();
    endInteractiveRenderSoon();
    return;
  }

  if (key === "-" || key === "_") {
    beginInteractiveRender();
    arcball.applyWheel(140);
    scheduleRender();
    updateHud();
    endInteractiveRenderSoon();
    return;
  }

  if (event.key === "Delete" || event.key === "Backspace") {
    event.preventDefault();
    removeSelectedObject();
    return;
  }

  if (key === "escape") {
    state.selectedId = null;
    syncCameraTargetToSelection();
    scheduleRender();
    updateHud();
    return;
  }

  if (key === "arrowleft") {
    event.preventDefault();
    nudgeSelectedObject(-36, 0);
    return;
  }

  if (key === "arrowright") {
    event.preventDefault();
    nudgeSelectedObject(36, 0);
    return;
  }

  if (key === "arrowup") {
    event.preventDefault();
    nudgeSelectedObject(0, -36);
    return;
  }

  if (key === "arrowdown") {
    event.preventDefault();
    nudgeSelectedObject(0, 36);
    return;
  }

  if (key === "g") {
    state.shading = "gouraud";
  } else if (key === "p") {
    state.shading = "phong";
  } else if (key === "n") {
    state.shading = "normals";
  } else if (key === "f") {
    state.shading = "wireframe";
  } else if (key === "r") {
    resetCameraView();
  } else {
    return;
  }

  scheduleRender();
  updateHud();
});

renderLightPanel();
updateHud();

await setTexture("tex4");
state.loadingMessage = "Loading teapot...";
updateHud();

await addObject("teapot");

async function loadAsset(assetId: string): Promise<LoadedAsset> {
  const cached = assetCache.get(assetId);

  if (cached) {
    return cached;
  }

  const preset = getPreset(assetId);

  let mesh: IndexedMesh;
  let fit: ReturnType<typeof createFitTransform>;

  switch (preset.source) {
    case "obj":
      mesh = await loadOBJ(preset.url ?? "", preset.label);
      fit = createFitTransform(preset.bounds);
      break;
    case "cube":
      mesh = createCubeMesh();
      fit = createFitTransform({
        kind: "sphere",
        center: mesh.center,
        radius: Math.max(mesh.radius, 0.001),
      });
      break;
    case "sphere":
      mesh = createSphereMesh();
      fit = createFitTransform({
        kind: "sphere",
        center: mesh.center,
        radius: Math.max(mesh.radius, 0.001),
      });
      break;
  }

  generateSphericalUVs(mesh, fit.center);

  const loaded = {
    preset,
    mesh,
    fit,
  };

  assetCache.set(assetId, loaded);
  return loaded;
}

async function addObject(assetId: string) {
  const asset = await loadAsset(assetId);
  const object: SceneObject = {
    id: nextObjectId,
    assetId,
    label: asset.preset.label,
    position: nextSpawnPosition(sceneObjects.length),
    scale: 1,
  };

  nextObjectId += 1;
  sceneObjects.push(object);
  state.selectedId = object.id;
  state.loadingMessage = "";
  resetCameraView();
  scheduleRender();
  updateHud();
}

async function setTexture(textureId: string) {
  const preset = getTexturePreset(textureId);
  const cached = textureCache.get(textureId);

  state.loadingMessage = `Loading ${preset.label}...`;
  updateHud();

  if (cached) {
    state.texture = cached;
    state.textureId = textureId;
    state.loadingMessage = "";
    scheduleRender();
    updateHud();
    return;
  }

  const texture = await loadTextureData(preset.url);
  textureCache.set(textureId, texture);
  state.texture = texture;
  state.textureId = textureId;
  state.loadingMessage = "";
  scheduleRender();
  updateHud();
}

function openModelFilePicker() {
  fileInput.click();
}

function getFirstUploadedFile(files: FileList | null) {
  if (!files) {
    return null;
  }

  return Array.from(files).find((file) => file.name.toLowerCase().endsWith(".obj")) ?? null;
}

async function importUploadedObject(file: File) {
  const fileName = file.name.trim() || "uploaded.obj";

  if (!fileName.toLowerCase().endsWith(".obj")) {
    window.alert(`Only OBJ files are supported right now. "${fileName}" was ignored.`);
    return;
  }

  state.loadingMessage = `Importing ${fileName}...`;
  updateHud();

  try {
    const text = await file.text();
    const mesh = parseOBJ(text, stripFileExtension(fileName));

    if (mesh.vertexCount === 0 || mesh.triangleCount === 0) {
      throw new Error("The OBJ file does not contain any triangles.");
    }

    const radius = Math.max(mesh.radius, 0.001);
    const fit = createFitTransform({
      kind: "sphere",
      center: mesh.center,
      radius,
    });

    generateSphericalUVs(mesh, fit.center);

    const assetId = createUploadedAssetId(fileName);
    assetCache.set(assetId, {
      preset: {
        id: assetId,
        label: stripFileExtension(fileName),
        source: "obj",
        url: fileName,
        bounds: {
          kind: "sphere",
          center: mesh.center,
          radius,
        },
      },
      mesh,
      fit,
    });

    await addObject(assetId);
  } catch (error) {
    state.loadingMessage = "";
    updateHud();
    window.alert(`Could not import "${fileName}". ${getErrorMessage(error)}`);
  }
}

function createUploadedAssetId(fileName: string) {
  const slug =
    stripFileExtension(fileName)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || "mesh";

  const assetId = `upload-${nextUploadedAssetId}-${slug}`;
  nextUploadedAssetId += 1;
  return assetId;
}

function stripFileExtension(fileName: string) {
  return fileName.replace(/\.[^.]+$/, "") || fileName;
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : "Unknown import error.";
}

function cloneLight(light: LightSettings): LightSettings {
  return {
    ...light,
    position: [...light.position],
  };
}

function renderLightPanel() {
  lightPanel.innerHTML = `
    <div class="light-card">
      <div class="light-title">Assets, Light & Shadows</div>
      <div class="light-actions">
        <button type="button" data-light-action="upload-obj">Upload Extra Figure</button>
      </div>
      <div class="light-hint">Sube un archivo .obj como figura extra, o usa U y drag & drop.</div>
      <label class="light-toggle">
        <input type="checkbox" data-light-toggle="shadow" ${state.light.shadowEnabled ? "checked" : ""}>
        <span>Enable shadows</span>
      </label>
      ${LIGHT_CONTROLS.map(
        (control) => `
          <label class="light-control">
            <span class="light-control-head">
              <span>${control.label}</span>
              <span class="light-value" data-light-value="${control.id}"></span>
            </span>
            <input
              class="light-slider"
              type="range"
              data-light-control="${control.id}"
              min="${control.min}"
              max="${control.max}"
              step="${control.step}"
              value="${control.getValue()}"
            >
          </label>
        `,
      ).join("")}
      <div class="light-actions">
        <button type="button" data-light-action="reset">Reset Light</button>
      </div>
      <div class="light-hint">Phong y Gouraud muestran mejor la luz y las sombras.</div>
    </div>
  `;

  syncLightPanel();
}

function syncLightPanel() {
  for (const control of LIGHT_CONTROLS) {
    const input = lightPanel.querySelector<HTMLInputElement>(`input[data-light-control="${control.id}"]`);
    const value = lightPanel.querySelector<HTMLElement>(`[data-light-value="${control.id}"]`);

    if (input) {
      input.value = String(control.getValue());
      input.disabled =
        !state.light.shadowEnabled &&
        (control.id === "shadow-strength" || control.id === "shadow-bias");
    }

    if (value) {
      value.textContent = formatLightValue(control.id, control.getValue());
    }
  }

  const toggle = lightPanel.querySelector<HTMLInputElement>('input[data-light-toggle="shadow"]');

  if (toggle) {
    toggle.checked = state.light.shadowEnabled;
  }
}

function formatLightValue(controlId: string, value: number) {
  if (controlId === "shininess") {
    return String(Math.round(value));
  }

  if (controlId === "shadow-bias") {
    return value.toFixed(4);
  }

  return value.toFixed(2);
}

function handleLightPanelInput(event: Event) {
  const input = event.target as HTMLInputElement | null;
  const controlId = input?.dataset.lightControl;

  if (!controlId) {
    return;
  }

  const control = LIGHT_CONTROLS.find((entry) => entry.id === controlId);

  if (!control) {
    return;
  }

  control.setValue(Number(input.value));
  syncLightPanel();
  beginInteractiveRender();
  scheduleRender();
  updateHud();
  endInteractiveRenderSoon();
}

function handleLightPanelChange(event: Event) {
  const input = event.target as HTMLInputElement | null;

  if (input?.dataset.lightToggle === "shadow") {
    state.light.shadowEnabled = input.checked;
    syncLightPanel();
    scheduleRender();
    updateHud();
  }
}

function handleLightPanelClick(event: Event) {
  const target = event.target as HTMLElement | null;
  const button = target?.closest("button[data-light-action]") as HTMLButtonElement | null;

  if (!button) {
    return;
  }

  if (button.dataset.lightAction === "upload-obj") {
    openModelFilePicker();
    return;
  }

  if (button.dataset.lightAction === "reset") {
    state.light = cloneLight(DEFAULT_LIGHT);
    syncLightPanel();
    scheduleRender();
    updateHud();
  }
}

function removeSelectedObject() {
  if (state.selectedId === null) {
    return;
  }

  const index = sceneObjects.findIndex((entry) => entry.id === state.selectedId);

  if (index === -1) {
    state.selectedId = null;
    updateHud();
    return;
  }

  sceneObjects.splice(index, 1);
  state.selectedId = sceneObjects.length > 0 ? sceneObjects[Math.max(0, sceneObjects.length - 1)].id : null;
  syncCameraTargetToSelection();
  scheduleRender();
  updateHud();
}

function handlePointerDown(event: PointerEvent) {
  beginInteractiveRender();
  dragState.startClientX = event.clientX;
  dragState.startClientY = event.clientY;
  dragState.lastClientX = event.clientX;
  dragState.lastClientY = event.clientY;
  dragState.moved = false;
  dragState.pointerDownPickId = renderer.pick(event.clientX, event.clientY);

  if (dragState.pointerDownPickId >= 0) {
    state.selectedId = dragState.pointerDownPickId;
    syncCameraTargetToSelection();

    if (beginObjectDrag(dragState.pointerDownPickId, event.clientX, event.clientY)) {
      dragState.mode = "move";
    } else {
      dragState.mode = "orbit";
      arcball.beginDrag(event.clientX, event.clientY);
    }
  } else {
    dragState.mode = "orbit";
    arcball.beginDrag(event.clientX, event.clientY);
  }

  appCanvas.setPointerCapture(event.pointerId);
  scheduleRender();
  updateHud();
}

function handlePointerMove(event: PointerEvent) {
  if (dragState.mode === "none") {
    return;
  }

  dragState.lastClientX = event.clientX;
  dragState.lastClientY = event.clientY;

  if (
    Math.abs(event.clientX - dragState.startClientX) > 4 ||
    Math.abs(event.clientY - dragState.startClientY) > 4
  ) {
    dragState.moved = true;
  }

  if (dragState.mode === "orbit") {
    arcball.dragTo(event.clientX, event.clientY);
  } else if (dragState.mode === "move") {
    moveSelectedObjectToPointer(event.clientX, event.clientY);
  }

  scheduleRender();
}

function handlePointerUp(event: PointerEvent) {
  if (dragState.mode === "orbit") {
    arcball.endDrag();

    if (!dragState.moved) {
      const pickedId = renderer.pick(event.clientX, event.clientY);
      state.selectedId = pickedId >= 0 ? pickedId : null;
      syncCameraTargetToSelection();
      updateHud();
    }
  }

  if (dragState.mode === "move" && dragState.pointerDownPickId >= 0) {
    state.selectedId = dragState.pointerDownPickId;
    syncCameraTargetToSelection();
    updateHud();
  }

  finishPointerInteraction(event.pointerId);
}

function handlePointerCancel(event: PointerEvent) {
  if (dragState.mode === "orbit") {
    arcball.endDrag();
  }

  finishPointerInteraction(event.pointerId);
}

function finishPointerInteraction(pointerId: number) {
  dragState.mode = "none";
  dragState.pointerDownPickId = -1;
  dragState.planePoint = [0, 0, 0];
  dragState.planeNormal = [0, 0, 1];
  dragState.positionOffset = [0, 0, 0];

  if (appCanvas.hasPointerCapture(pointerId)) {
    appCanvas.releasePointerCapture(pointerId);
  }

  scheduleRender();
  endInteractiveRenderSoon();
}

function scheduleRender() {
  if (pendingRender) {
    return;
  }

  pendingRender = true;

  requestAnimationFrame(() => {
    pendingRender = false;
    renderScene();
  });
}

function renderScene() {
  const texture = state.texture;

  if (!texture) {
    return;
  }

  const camera = getCameraState();
  const focus = getCameraFocus();
  const renderObjects: RenderObject[] = [];

  for (const object of sceneObjects) {
    const asset = assetCache.get(object.assetId);

    if (!asset) {
      continue;
    }

    const scale = mat4.scaling(object.scale, object.scale, object.scale);
    const translation = mat4.translation(object.position[0], object.position[1], object.position[2]);
    const modelMatrix = mat4.multiply(translation, mat4.multiply(scale, asset.fit.matrix));

    renderObjects.push({
      id: object.id,
      mesh: asset.mesh,
      modelMatrix,
    });
  }

  const totalTriangles = renderObjects.reduce((sum, entry) => sum + entry.mesh.triangleCount, 0);
  const quality = getQualitySettings(totalTriangles);

  const renderLight: LightSettings = {
    ...state.light,
    position: [...state.light.position],
    shadowEnabled: state.interactive ? false : state.light.shadowEnabled,
  };

  const stats = renderer.render({
    objects: renderObjects,
    viewMatrix: camera.viewMatrix,
    projectionMatrix: camera.projectionMatrix,
    cameraPosition: camera.position,
    near: camera.near,
    far: camera.far,
    texture,
    shading: state.shading,
    light: renderLight,
    shadowCenter: focus.target,
    shadowRadius: Math.max(2, focus.sceneRadius),
    clearColor:
      state.shading === "wireframe" ? [246, 244, 237] : [18, 26, 37],
    resolutionScale: quality.scale,
    maxPixels: quality.maxPixels,
    shadowMapSize: state.interactive ? 256 : 1024,
    selectedObjectId: state.selectedId,
  });

  updateHud(stats);
}

function beginObjectDrag(objectId: number, clientX: number, clientY: number) {
  const selected = sceneObjects.find((entry) => entry.id === objectId);

  if (!selected) {
    return false;
  }

  const camera = getCameraState();
  const ray = getPointerRay(clientX, clientY, camera);

  if (!ray) {
    return false;
  }

  const planeNormal = camera.forward;
  const hitPoint = intersectRayPlane(ray.origin, ray.direction, selected.position, planeNormal);

  if (!hitPoint) {
    return false;
  }

  dragState.planePoint = [...selected.position];
  dragState.planeNormal = [...planeNormal];
  dragState.positionOffset = vec3.sub(selected.position, hitPoint);
  return true;
}

function moveSelectedObjectToPointer(clientX: number, clientY: number) {
  if (state.selectedId === null) {
    return;
  }

  const selected = sceneObjects.find((entry) => entry.id === state.selectedId);

  if (!selected) {
    return;
  }

  const camera = getCameraState();
  const ray = getPointerRay(clientX, clientY, camera);

  if (!ray) {
    return;
  }

  const hitPoint = intersectRayPlane(
    ray.origin,
    ray.direction,
    dragState.planePoint,
    dragState.planeNormal,
  );

  if (!hitPoint) {
    return;
  }

  selected.position = vec3.add(hitPoint, dragState.positionOffset);
}

function translateSelectedObjectByScreenPixels(deltaX: number, deltaY: number) {
  if (state.selectedId === null) {
    return;
  }

  const selected = sceneObjects.find((entry) => entry.id === state.selectedId);

  if (!selected) {
    return;
  }

  const camera = getCameraState();
  const depth = Math.max(0.35, vec3.length(vec3.sub(camera.position, selected.position)));
  const halfHeight = Math.tan(camera.fovY * 0.5) * depth;
  const unitsPerPixelY = (halfHeight * 2) / Math.max(1, appCanvas.clientHeight);
  const unitsPerPixelX = (halfHeight * 2 * camera.aspect) / Math.max(1, appCanvas.clientWidth);
  const deltaWorld = vec3.add(
    vec3.scale(camera.right, deltaX * unitsPerPixelX),
    vec3.scale(camera.up, -deltaY * unitsPerPixelY),
  );

  selected.position = vec3.add(selected.position, deltaWorld);
}

function nudgeSelectedObject(deltaX: number, deltaY: number) {
  beginInteractiveRender();
  translateSelectedObjectByScreenPixels(deltaX, deltaY);
  scheduleRender();
  updateHud();
  endInteractiveRenderSoon();
}

function resetCameraView() {
  arcball.reset();
  syncCameraTargetToSelection();
}

function getCameraState() {
  const aspect = Math.max(1, appCanvas.clientWidth) / Math.max(1, appCanvas.clientHeight);
  const fovY = Math.PI / 3;
  const halfVertical = fovY * 0.5;
  const halfHorizontal = Math.atan(Math.tan(halfVertical) * aspect);
  const fitHalfAngle = Math.max(0.1, Math.min(halfVertical, halfHorizontal));
  const { target, focusRadius, sceneRadius } = getCameraFocus();
  const baseDistance = (Math.max(1, focusRadius) / Math.sin(fitHalfAngle)) * 1.08;
  const cameraDistance = baseDistance / arcball.zoom;
  const near = 0.08;
  const far = cameraDistance + sceneRadius + 8;
  const cameraRotationMatrix = mat4.fromQuat(arcball.rotation);
  const orbitOffset = mat4.transformPoint(cameraRotationMatrix, [0, cameraDistance * 0.18, cameraDistance]);
  const position = vec3.add(target, orbitOffset);
  const up = vec3.normalize(mat4.transformVector(cameraRotationMatrix, [0, 1, 0]));
  const viewMatrix = mat4.lookAt(position, target, up);
  const projectionMatrix = mat4.perspective(fovY, aspect, near, far);
  const forward = vec3.normalize(vec3.sub(target, position));
  const right = vec3.normalize(vec3.cross(forward, up));

  return {
    aspect,
    fovY,
    cameraDistance,
    near,
    far,
    target,
    position,
    up,
    forward,
    right,
    viewMatrix,
    projectionMatrix,
  };
}

function computeSceneRadius() {
  if (sceneObjects.length === 0) {
    return 1.5;
  }

  let radius = 1.5;

  for (const object of sceneObjects) {
    radius = Math.max(radius, vec3.length(object.position) + object.scale);
  }

  return radius;
}

function getCameraFocus() {
  const selected = getSelectedObject();

  if (selected) {
    return {
      target: state.cameraTarget,
      focusRadius: Math.max(1, selected.scale),
      sceneRadius: computeSceneExtent(state.cameraTarget),
    };
  }

  return {
    target: state.cameraTarget,
    focusRadius: Math.max(1.5, computeSceneRadius()),
    sceneRadius: computeSceneExtent(state.cameraTarget),
  };
}

function computeSceneExtent(target: Vec3) {
  if (sceneObjects.length === 0) {
    return 1.5;
  }

  let extent = 1.5;

  for (const object of sceneObjects) {
    extent = Math.max(extent, vec3.length(vec3.sub(object.position, target)) + object.scale);
  }

  return extent;
}

function getQualitySettings(totalTriangles: number) {
  if (state.interactive) {
    if (totalTriangles > 50000) {
      return { scale: 0.32, maxPixels: 130000 };
    }

    if (totalTriangles > 12000) {
      return { scale: 0.48, maxPixels: 220000 };
    }

    return { scale: 0.72, maxPixels: 380000 };
  }

  if (totalTriangles > 50000) {
    return { scale: 0.62, maxPixels: 420000 };
  }

  if (totalTriangles > 12000) {
    return { scale: 0.85, maxPixels: 720000 };
  }

  return { scale: 1.15, maxPixels: 1200000 };
}

function updateHud(stats?: RenderStats) {
  const selected = getSelectedObject();
  const selectedLabel = selected ? `${selected.label} #${selected.id}` : "None";
  const textureLabel = getTexturePreset(state.textureId).label;
  const lightPosition = state.light.position.map((value) => value.toFixed(1)).join(", ");
  const statsInfo = stats
    ? `<div><strong>Objects:</strong> ${sceneObjects.length}</div>
       <div><strong>Texture:</strong> ${textureLabel}</div>
       <div><strong>Light:</strong> ${lightPosition}</div>
       <div><strong>Shadows:</strong> ${state.light.shadowEnabled ? "On" : "Off"}</div>
       <div><strong>Triangles:</strong> ${stats.submittedTriangles.toLocaleString()}</div>
       <div><strong>Rasterized:</strong> ${stats.rasterizedTriangles.toLocaleString()}</div>
       <div><strong>Resolution:</strong> ${stats.width} x ${stats.height}</div>
       <div><strong>Render:</strong> ${stats.renderTimeMs.toFixed(1)} ms</div>`
    : `<div class="hud-status">${state.loadingMessage}</div>`;

  hud.innerHTML = `
    <div class="hud-card">
      <div class="hud-title">Scene</div>
      <div><strong>Mode:</strong> ${getShadingLabel(state.shading)}</div>
      <div><strong>Selected:</strong> ${selectedLabel}</div>
      <div><strong>Quality:</strong> ${state.interactive ? "Interactive" : "Final"}</div>
      ${statsInfo}
    </div>
    <div class="hud-card">
      <div class="hud-title">Trackpad</div>
      <div>Click: select object</div>
      <div>Drag object: move object</div>
      <div>Drag background: arcball</div>
      <div>Two-finger scroll: arcball</div>
      <div>Pinch or Alt+scroll: zoom</div>
      <div>Arrow keys: nudge selected</div>
      <div>U or Upload OBJ: import model</div>
      <div>H/J/K/L: change texture</div>
      <div>G/P/N/F: shading modes</div>
      <div>Light panel: move light and shadows</div>
      <div>N: normal buffer view</div>
      <div>F: wireframe view</div>
    </div>
  `;

  renderToolbar();
}

function renderToolbar() {
  const selected = getSelectedObject();
  const disableSelected = selected ? "" : "disabled";
  const textureLabel = getTexturePreset(state.textureId).label;

  toolbar.innerHTML = `
    <div class="toolbar-card">
      <div class="toolbar-title">Quick Actions</div>
      <div class="toolbar-row">
        <button data-action="add-teapot">Teapot</button>
        <button data-action="add-beacon">Beacon</button>
        <button data-action="add-cube">Cube</button>
      </div>
      <div class="toolbar-row">
        <button data-action="add-sphere">Sphere</button>
        <button data-action="delete-selected" ${disableSelected}>Delete</button>
        <button data-action="clear-selection" ${disableSelected}>Unselect</button>
      </div>
      <div class="toolbar-grid toolbar-grid-single">
        <button data-action="upload-obj" class="toolbar-upload">Upload OBJ</button>
      </div>
      <div class="toolbar-row">
        <button data-action="zoom-out">-</button>
        <button data-action="zoom-in">+</button>
        <button data-action="reset-view">Reset</button>
      </div>
      <div class="toolbar-grid">
        <button data-action="texture-tex1">Tex H</button>
        <button data-action="texture-tex2">Tex J</button>
        <button data-action="texture-tex3">Tex K</button>
        <button data-action="texture-tex4">Tex L</button>
      </div>
      <div class="toolbar-grid">
        <button data-action="nudge-up" ${disableSelected}>Up</button>
        <button data-action="nudge-left" ${disableSelected}>Left</button>
        <button data-action="nudge-down" ${disableSelected}>Down</button>
        <button data-action="nudge-right" ${disableSelected}>Right</button>
      </div>
      <div class="toolbar-hint">
        H/J/K/L change the texture.
      </div>
      <div class="toolbar-hint">
        U or drop an OBJ file on the canvas to add your own model.
      </div>
      <div class="toolbar-hint">
        G/P/N/F change the render mode. N shows normal buffer and F shows wireframe.
      </div>
      <div class="toolbar-hint">
        Current texture: ${textureLabel}
      </div>
    </div>
  `;
}

async function handleToolbarAction(action: string) {
  switch (action) {
    case "add-teapot":
      await addObject("teapot");
      return;
    case "add-beacon":
      await addObject("beacon");
      return;
    case "add-cube":
      await addObject("cube");
      return;
    case "add-sphere":
      await addObject("sphere");
      return;
    case "upload-obj":
      openModelFilePicker();
      return;
    case "texture-tex1":
      await setTexture("tex1");
      return;
    case "texture-tex2":
      await setTexture("tex2");
      return;
    case "texture-tex3":
      await setTexture("tex3");
      return;
    case "texture-tex4":
      await setTexture("tex4");
      return;
    case "delete-selected":
      removeSelectedObject();
      return;
    case "clear-selection":
      state.selectedId = null;
      syncCameraTargetToSelection();
      scheduleRender();
      updateHud();
      return;
    case "zoom-in":
      beginInteractiveRender();
      arcball.applyWheel(-140);
      scheduleRender();
      updateHud();
      endInteractiveRenderSoon();
      return;
    case "zoom-out":
      beginInteractiveRender();
      arcball.applyWheel(140);
      scheduleRender();
      updateHud();
      endInteractiveRenderSoon();
      return;
    case "reset-view":
      resetCameraView();
      scheduleRender();
      updateHud();
      return;
    case "nudge-up":
      nudgeSelectedObject(0, -36);
      return;
    case "nudge-left":
      nudgeSelectedObject(-36, 0);
      return;
    case "nudge-down":
      nudgeSelectedObject(0, 36);
      return;
    case "nudge-right":
      nudgeSelectedObject(36, 0);
      return;
    default:
      return;
  }
}

function getSelectedObject() {
  return state.selectedId === null
    ? null
    : sceneObjects.find((entry) => entry.id === state.selectedId) ?? null;
}

function beginInteractiveRender() {
  const changed = !state.interactive;
  state.interactive = true;
  window.clearTimeout(interactiveTimer);

  if (changed) {
    scheduleRender();
    updateHud();
  }
}

function endInteractiveRenderSoon() {
  window.clearTimeout(interactiveTimer);
  interactiveTimer = window.setTimeout(() => {
    state.interactive = false;
    scheduleRender();
    updateHud();
  }, 140);
}

function getPreset(assetId: string) {
  const preset = MODEL_PRESETS.find((entry) => entry.id === assetId);

  if (!preset) {
    throw new Error(`Unknown asset: ${assetId}`);
  }

  return preset;
}

function getTexturePreset(textureId: string) {
  const preset = TEXTURE_PRESETS.find((entry) => entry.id === textureId);

  if (!preset) {
    throw new Error(`Unknown texture: ${textureId}`);
  }

  return preset;
}

function syncCameraTargetToSelection() {
  const selected = getSelectedObject();

  if (selected) {
    state.cameraTarget = [...selected.position];
    return;
  }

  if (sceneObjects.length === 0) {
    state.cameraTarget = [0, 0, 0];
    return;
  }

  const center = sceneObjects.reduce<Vec3>(
    (sum, object) => [sum[0] + object.position[0], sum[1] + object.position[1], sum[2] + object.position[2]],
    [0, 0, 0],
  );

  state.cameraTarget = [
    center[0] / sceneObjects.length,
    center[1] / sceneObjects.length,
    center[2] / sceneObjects.length,
  ];
}

function getPointerRay(clientX: number, clientY: number, camera: ReturnType<typeof getCameraState>) {
  const rect = appCanvas.getBoundingClientRect();

  if (rect.width <= 0 || rect.height <= 0) {
    return null;
  }

  const normalizedX = ((clientX - rect.left) / rect.width) * 2 - 1;
  const normalizedY = 1 - ((clientY - rect.top) / rect.height) * 2;
  const tanHalfY = Math.tan(camera.fovY * 0.5);
  const tanHalfX = tanHalfY * camera.aspect;

  const direction = vec3.normalize(
    vec3.add(
      camera.forward,
      vec3.add(
        vec3.scale(camera.right, normalizedX * tanHalfX),
        vec3.scale(camera.up, normalizedY * tanHalfY),
      ),
    ),
  );

  return {
    origin: camera.position,
    direction,
  };
}

function intersectRayPlane(
  origin: Vec3,
  direction: Vec3,
  planePoint: Vec3,
  planeNormal: Vec3,
) {
  const denominator = vec3.dot(direction, planeNormal);

  if (Math.abs(denominator) < 1e-5) {
    return null;
  }

  const distance = vec3.dot(vec3.sub(planePoint, origin), planeNormal) / denominator;

  if (distance <= 0) {
    return null;
  }

  return vec3.add(origin, vec3.scale(direction, distance));
}

function nextSpawnPosition(index: number): Vec3 {
  if (index === 0) {
    return [0, 0, 0];
  }

  const step = Math.ceil(index / 2);
  const direction = index % 2 === 0 ? 1 : -1;
  return [direction * step * 2.4, 0, 0];
}

function createPrimitiveMesh(name: string, positions: number[], indices: number[]): IndexedMesh {
  const positionData = Float32Array.from(positions);
  const mesh: IndexedMesh = {
    name,
    positions: positionData,
    normals: new Float32Array(positionData.length),
    uvs: new Float32Array((positionData.length / 3) * 2),
    indices: Uint32Array.from(indices),
    faceNormals: new Float32Array(indices.length),
    vertexCount: positionData.length / 3,
    triangleCount: indices.length / 3,
    ...computeMeshBounds(positionData),
  };

  recomputeNormals(mesh);
  generateSphericalUVs(mesh, mesh.center);
  return mesh;
}

function createCubeMesh() {
  return createPrimitiveMesh(
    "Cube",
    [
      -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1,
      -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1,
      -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1,
      1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1,
      -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1,
      -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1,
    ],
    [
      0, 1, 2, 0, 2, 3,
      4, 5, 6, 4, 6, 7,
      8, 9, 10, 8, 10, 11,
      12, 13, 14, 12, 14, 15,
      16, 17, 18, 16, 18, 19,
      20, 21, 22, 20, 22, 23,
    ],
  );
}

function createSphereMesh(latitudeSegments = 20, longitudeSegments = 28) {
  const positions: number[] = [];
  const indices: number[] = [];

  for (let lat = 0; lat <= latitudeSegments; lat += 1) {
    const v = lat / latitudeSegments;
    const theta = v * Math.PI;
    const sinTheta = Math.sin(theta);
    const cosTheta = Math.cos(theta);

    for (let lon = 0; lon <= longitudeSegments; lon += 1) {
      const u = lon / longitudeSegments;
      const phi = u * Math.PI * 2;
      const sinPhi = Math.sin(phi);
      const cosPhi = Math.cos(phi);

      positions.push(
        sinTheta * cosPhi,
        cosTheta,
        sinTheta * sinPhi,
      );
    }
  }

  const stride = longitudeSegments + 1;

  for (let lat = 0; lat < latitudeSegments; lat += 1) {
    for (let lon = 0; lon < longitudeSegments; lon += 1) {
      const a = lat * stride + lon;
      const b = a + stride;
      const c = a + 1;
      const d = b + 1;

      if (lat > 0) {
        indices.push(a, b, c);
      }

      if (lat < latitudeSegments - 1) {
        indices.push(c, b, d);
      }
    }
  }

  return createPrimitiveMesh("Sphere", positions, indices);
}

function getShadingLabel(mode: ShadingMode): string {
  switch (mode) {
    case "gouraud":
      return "Gouraud shading";
    case "phong":
      return "Phong shading";
    case "normals":
      return "Normal buffer";
    case "wireframe":
      return "Wireframe";
  }
}