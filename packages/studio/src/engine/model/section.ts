// engine/model/section.ts — the Section tool: slice the model open.

import * as THREE from 'three';
import { RENDER_UNITS_PER_MICRON } from './part';

export type SectionAxis = 'x' | 'y' | 'z';

export interface SectionPlacement {
  axis: SectionAxis;
  /** How deep the cut goes, in µm, measured from the +axis face of the model. */
  depthMicrons: number;
}

const AXIS_VECTORS: Record<SectionAxis, THREE.Vector3> = {
  x: new THREE.Vector3(1, 0, 0),
  y: new THREE.Vector3(0, 1, 0),
  z: new THREE.Vector3(0, 0, 1),
};

// The handle is drawn slightly larger than the model's cross-section so its
// rim is grabbable even when the cut is flush with a face.
const HANDLE_OVERHANG_RATIO = 1.15;
const HANDLE_OPACITY = 0.12;

export interface SectionController {
  /** Add once to the scene; contains the (initially hidden) drag handle. */
  group: THREE.Group;
  /**
   * Apply a placement. Returns the clipping planes the materials should use —
   * one plane when active, empty when off (pass straight to setClippingPlanes).
   */
  apply(placement: SectionPlacement | null, modelBoundsMicrons: THREE.Box3): THREE.Plane[];
  /** True while a drag on the handle is in progress (viewport must not orbit). */
  isDragging(): boolean;
  /** Wire pointer events for the drag handle. Call once after construction. */
  attach(canvas: HTMLElement, camera: THREE.Camera, onDepthChange: (depthMicrons: number) => void): void;
  dispose(): void;
}

export function createSectionController(): SectionController {
  const group = new THREE.Group();
  group.name = 'section-tool';

  const handleMaterial = new THREE.MeshBasicMaterial({
    color: 0x3b82f6, // accent-ish; the fill is faint, the meaning is "cut here"
    transparent: true,
    opacity: HANDLE_OPACITY,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const handleMesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), handleMaterial);
  handleMesh.name = 'section-drag-handle';
  handleMesh.visible = false;
  group.add(handleMesh);

  const clippingPlane = new THREE.Plane();
  let activePlacement: SectionPlacement | null = null;
  let boundsMicrons = new THREE.Box3();
  let dragging = false;

  function apply(
    placement: SectionPlacement | null,
    modelBoundsMicrons: THREE.Box3,
  ): THREE.Plane[] {
    activePlacement = placement;
    boundsMicrons = modelBoundsMicrons.clone();
    if (!placement || modelBoundsMicrons.isEmpty()) {
      handleMesh.visible = false;
      return [];
    }

    const { axis } = placement;
    const axisDirection = AXIS_VECTORS[axis];
    const cutMicrons = cutCoordinateMicrons(placement);
    const cutRender = cutMicrons * RENDER_UNITS_PER_MICRON;

    // Normal points AGAINST the axis: geometry beyond the cut (toward +axis)
    // is discarded, geometry before it is kept.
    clippingPlane.normal.copy(axisDirection).negate();
    clippingPlane.constant = cutRender;

    positionHandle(axis, cutRender);
    handleMesh.visible = true;
    return [clippingPlane];
  }

  /** Clamp the typed depth to the model, convert to an absolute coordinate. */
  function cutCoordinateMicrons({ axis, depthMicrons }: SectionPlacement): number {
    const max = boundsMicrons.max[axis];
    const min = boundsMicrons.min[axis];
    return THREE.MathUtils.clamp(max - depthMicrons, min, max);
  }

  function positionHandle(axis: SectionAxis, cutRender: number): void {
    const sizeRender = boundsMicrons
      .getSize(new THREE.Vector3())
      .multiplyScalar(RENDER_UNITS_PER_MICRON * HANDLE_OVERHANG_RATIO);
    const centerRender = boundsMicrons
      .getCenter(new THREE.Vector3())
      .multiplyScalar(RENDER_UNITS_PER_MICRON);

    // PlaneGeometry faces +Z by default; rotate it to face the cut axis.
    handleMesh.rotation.set(0, 0, 0);
    if (axis === 'x') {
      handleMesh.rotation.y = Math.PI / 2;
      handleMesh.scale.set(sizeRender.z, sizeRender.y, 1);
      handleMesh.position.set(cutRender, centerRender.y, centerRender.z);
    } else if (axis === 'y') {
      handleMesh.rotation.x = -Math.PI / 2;
      handleMesh.scale.set(sizeRender.x, sizeRender.z, 1);
      handleMesh.position.set(centerRender.x, cutRender, centerRender.z);
    } else {
      handleMesh.scale.set(sizeRender.x, sizeRender.y, 1);
      handleMesh.position.set(centerRender.x, centerRender.y, cutRender);
    }
  }

  // ── drag: pixels moved along the axis's screen direction → µm of depth ──
  const _raycaster = new THREE.Raycaster();
  const _ndc = new THREE.Vector2();

  function attach(
    canvas: HTMLElement,
    camera: THREE.Camera,
    onDepthChange: (depthMicrons: number) => void,
  ): void {
    let lastClient = { x: 0, y: 0 };

    function handleHit(event: PointerEvent): boolean {
      const rect = canvas.getBoundingClientRect();
      _ndc.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      _raycaster.setFromCamera(_ndc, camera);
      return _raycaster.intersectObject(handleMesh).length > 0;
    }

    /** How many µm of depth one dragged pixel is worth, along the cut axis. */
    function micronsPerPixelAlongAxis(axis: SectionAxis): { x: number; y: number } {
      const rect = canvas.getBoundingClientRect();
      const center = boundsMicrons
        .getCenter(new THREE.Vector3())
        .multiplyScalar(RENDER_UNITS_PER_MICRON);
      const alongAxis = center
        .clone()
        .add(AXIS_VECTORS[axis].clone().multiplyScalar(RENDER_UNITS_PER_MICRON * 1000)); // +1000 µm probe

      const centerScreen = center.project(camera);
      const alongScreen = alongAxis.project(camera);
      const dxPx = ((alongScreen.x - centerScreen.x) / 2) * rect.width;
      const dyPx = -((alongScreen.y - centerScreen.y) / 2) * rect.height;
      const pixelsPer1000Microns = Math.hypot(dxPx, dyPx) || 1;
      // Unit vector of the axis on screen, scaled to µm-per-pixel. Dragging
      // toward +axis on screen should DECREASE depth (the cut follows the hand).
      return {
        x: (dxPx / pixelsPer1000Microns) * (1000 / pixelsPer1000Microns),
        y: (dyPx / pixelsPer1000Microns) * (1000 / pixelsPer1000Microns),
      };
    }

    function onPointerDown(event: PointerEvent): void {
      if (event.button !== 0 || !handleMesh.visible || !activePlacement) return;
      if (!handleHit(event)) return;
      dragging = true;
      lastClient = { x: event.clientX, y: event.clientY };
      canvas.setPointerCapture(event.pointerId);
      // Stop the orbit controls (which listen on the same canvas) from
      // treating this drag as a rotation.
      event.stopImmediatePropagation();
    }

    function onPointerMove(event: PointerEvent): void {
      if (!dragging || !activePlacement) return;
      const perPixel = micronsPerPixelAlongAxis(activePlacement.axis);
      const movedMicrons =
        (event.clientX - lastClient.x) * perPixel.x +
        (event.clientY - lastClient.y) * perPixel.y;
      lastClient = { x: event.clientX, y: event.clientY };

      const spanMicrons =
        boundsMicrons.max[activePlacement.axis] - boundsMicrons.min[activePlacement.axis];
      const nextDepth = THREE.MathUtils.clamp(
        activePlacement.depthMicrons - movedMicrons,
        0,
        spanMicrons,
      );
      onDepthChange(nextDepth);
      event.stopImmediatePropagation();
    }

    function onPointerUp(): void {
      dragging = false;
    }

    // Capture phase: the handle must win over the orbit controls on the same canvas.
    canvas.addEventListener('pointerdown', onPointerDown, { capture: true });
    canvas.addEventListener('pointermove', onPointerMove, { capture: true });
    canvas.addEventListener('pointerup', onPointerUp, { capture: true });
    detachListeners = () => {
      canvas.removeEventListener('pointerdown', onPointerDown, { capture: true });
      canvas.removeEventListener('pointermove', onPointerMove, { capture: true });
      canvas.removeEventListener('pointerup', onPointerUp, { capture: true });
    };
  }

  let detachListeners = () => {};

  return {
    group,
    apply,
    isDragging: () => dragging,
    attach,
    dispose() {
      detachListeners();
      handleMesh.geometry.dispose();
      handleMaterial.dispose();
      group.removeFromParent();
    },
  };
}
