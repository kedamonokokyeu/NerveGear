// engine/model/chipScene.ts
// Turns Parts into the 3D scene, and clean up after.

import * as THREE from 'three';
import type { Part } from './part';
import { RENDER_UNITS_PER_MICRON } from './part';
import { materialInfo } from './materials';

// Threshold angle between neighboring triangles where an edge should be drawn.
// If two faces meet at more than this threshold, it's a visible crease. 
// Helps ignore tiny seams on curved surfaces.
export const FEATURE_EDGE_ANGLE_DEGREES = 35;

// Chip's lid will be partially transparent so users can see internal stack underneath.
const LID_OPACITY = 0.35;

// 3D scene object --- represents chip model inside viewer (parts, materials, rendering, etc.)
export interface ChipScene {
  root: THREE.Group; // Top-level Three.js group containing all the chip's meshes (scene.add(chipScene.root);)
  meshByPartId: Map<string, THREE.Mesh>; // Lookup table to find/manipulate parts by ID.
  setPartMaterial(partId: string, materialKey: string): void; // Lets you recolor a part.
  setEdgeColor(cssColor: string): void; // Edge lines matches UI theme.
  setClippingPlanes(planes: THREE.Plane[]): void; // Applies section cuts for visualization.
  dispose(): void; // Clean GPU memory when scene is closed.
}

export function buildChipScene(parts: Part[]): ChipScene {
  const root = new THREE.Group(); // THREE.Group is a sub-container within the THREE.Scene world.
  root.name = 'chip-model';
  // Geometry in micrometers, so convert into millimeters to allow for easier calculations for camera math, etc.
  root.scale.setScalar(RENDER_UNITS_PER_MICRON);

  const meshByPartId = new Map<string, THREE.Mesh>(); // lookup table for part ID

  const partMaterials: THREE.MeshLambertMaterial[] = []; // Separate material instance for each part (for clipping).

  const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x000000 }); // Black line material to draw feature edges.

  const edgeGeometries: THREE.BufferGeometry[] = []; // Stores geometries for edge lines.

  for (const part of parts) {
    const material = makePartMaterial(part.materialKey);
    partMaterials.push(material);

    const mesh = new THREE.Mesh(part.geometry, material);
    mesh.name = part.name;
    // THREE.Mesh has a built-in userData object (stores custom info).
    // Save the ID there so during raycast (3D "click" test) the program knows which part was hit.
    mesh.userData.partId = part.id;

    const edges = new THREE.EdgesGeometry(part.geometry, FEATURE_EDGE_ANGLE_DEGREES); // Differentiates edges and insignificant creases.
    edgeGeometries.push(edges);
    const outline = new THREE.LineSegments(edges, edgeMaterial); // Edge geometry outlines.
    outline.name = `${part.name}-edges`;
    outline.raycast = () => {}; // If outline is clicked, nothing should occur. Clicks should only affect actual parts.
    mesh.add(outline); // Outlines are child objects of the mesh --- if mesh is rerendered, so are the outlines.

    root.add(mesh);
    meshByPartId.set(part.id, mesh);
  }

  function setPartMaterial(partId: string, materialKey: string): void {
    const mesh = meshByPartId.get(partId);
    if (!mesh) return;
    const material = mesh.material as THREE.MeshLambertMaterial;
    const info = materialInfo(materialKey);
    material.color.set(info.colorHex);
    material.transparent = materialKey === 'lid'; // Transparent only if part = lid.
    material.opacity = materialKey === 'lid' ? LID_OPACITY : 1;
    material.needsUpdate = true; // Flag for re-rendering in render loop.
  }

  function setEdgeColor(cssColor: string): void {
    edgeMaterial.color.set(cssColor.trim());
  }

  function setClippingPlanes(planes: THREE.Plane[]): void {
    for (const material of partMaterials) {
      material.clippingPlanes = planes;
      // A clipped-open box shows its hollow inside; render back faces so the
      // interior reads as solid walls instead of vanishing.
      material.side = planes.length > 0 ? THREE.DoubleSide : THREE.FrontSide;
      material.needsUpdate = true;
    }
    edgeMaterial.clippingPlanes = planes;
    edgeMaterial.needsUpdate = true;
  }

  function dispose(): void {
    for (const mesh of meshByPartId.values()) mesh.geometry.dispose();
    for (const edges of edgeGeometries) edges.dispose();
    for (const material of partMaterials) material.dispose();
    edgeMaterial.dispose();
    root.removeFromParent();
    meshByPartId.clear();
  }

  return { root, meshByPartId, setPartMaterial, setEdgeColor, setClippingPlanes, dispose };
}

function makePartMaterial(materialKey: string): THREE.MeshLambertMaterial {
  const info = materialInfo(materialKey);
  return new THREE.MeshLambertMaterial({ // Lambert material for silicon aesthetic
    color: info.colorHex,
    transparent: materialKey === 'lid',
    opacity: materialKey === 'lid' ? LID_OPACITY : 1, // Solid layers except for the lid.
  });
}
