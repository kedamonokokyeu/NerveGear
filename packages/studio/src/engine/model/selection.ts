// engine/model/selection.ts — click a part, see it highlighted.

import * as THREE from 'three';
import type { ChipScene } from './chipScene';

const _raycaster = new THREE.Raycaster();
const _pointerNdc = new THREE.Vector2();

/**
 * Which part is under this client-pixel position? Returns the part id, or
 * null for empty space. Parts opt in by carrying userData.partId (chipScene
 * sets that); anything else in the scene — grids, handles — is ignored.
 */
export function pickPartAt(
  clientX: number,
  clientY: number,
  canvas: HTMLElement,
  camera: THREE.Camera,
  chipRoot: THREE.Object3D,
): string | null {
  const rect = canvas.getBoundingClientRect();
  _pointerNdc.set(
    ((clientX - rect.left) / rect.width) * 2 - 1,
    -((clientY - rect.top) / rect.height) * 2 + 1,
  );
  _raycaster.setFromCamera(_pointerNdc, camera);

  for (const hit of _raycaster.intersectObjects(chipRoot.children, true)) {
    const partId = findPartId(hit.object);
    if (partId) return partId;
  }
  return null;
}

/** Walk up from the hit object to the mesh that carries the part id. */
function findPartId(hitObject: THREE.Object3D): string | null {
  let node: THREE.Object3D | null = hitObject;
  while (node) {
    if (typeof node.userData.partId === 'string') return node.userData.partId;
    node = node.parent;
  }
  return null;
}

export interface SelectionHighlighter {
  /** Show the accent outline on this part (null clears it). */
  highlight(partId: string | null): void;
  /** Keep the outline consistent with the Section tool's cut. */
  setClippingPlanes(planes: THREE.Plane[]): void;
  setAccentColor(cssColor: string): void;
  dispose(): void;
}

// Drawn after opaque geometry and without depth-testing, so the outline stays
// visible even where the part is hidden behind its neighbours — that "x-ray"
// hint is what makes thin buried layers findable.
const OUTLINE_RENDER_ORDER = 10;

export function createSelectionHighlighter(
  chipScene: ChipScene,
  accentColorHex: string,
): SelectionHighlighter {
  const accentMaterial = new THREE.LineBasicMaterial({
    color: accentColorHex,
    depthTest: false,
  });
  // One overlay per part, created the first time that part is selected.
  // Overlays REUSE the part's existing edge geometry — zero extra GPU memory.
  const overlayByPartId = new Map<string, THREE.LineSegments>();
  let visibleOverlay: THREE.LineSegments | null = null;

  function overlayFor(partId: string): THREE.LineSegments | null {
    const existing = overlayByPartId.get(partId);
    if (existing) return existing;

    const mesh = chipScene.meshByPartId.get(partId);
    const edges = mesh?.children.find(
      (child): child is THREE.LineSegments => (child as THREE.LineSegments).isLineSegments,
    );
    if (!mesh || !edges) return null;

    const overlay = new THREE.LineSegments(edges.geometry, accentMaterial);
    overlay.renderOrder = OUTLINE_RENDER_ORDER;
    overlay.raycast = () => {}; // the outline itself is never clickable
    overlay.visible = false;
    mesh.add(overlay);
    overlayByPartId.set(partId, overlay);
    return overlay;
  }

  return {
    highlight(partId) {
      if (visibleOverlay) visibleOverlay.visible = false;
      visibleOverlay = partId ? overlayFor(partId) : null;
      if (visibleOverlay) visibleOverlay.visible = true;
    },
    setClippingPlanes(planes) {
      accentMaterial.clippingPlanes = planes;
      accentMaterial.needsUpdate = true;
    },
    setAccentColor(cssColor) {
      accentMaterial.color.set(cssColor.trim());
    },
    dispose() {
      // Overlay geometries belong to chipScene (shared), so only the material
      // and the scene-graph nodes are ours to clean up.
      for (const overlay of overlayByPartId.values()) overlay.removeFromParent();
      overlayByPartId.clear();
      accentMaterial.dispose();
      visibleOverlay = null;
    },
  };
}
