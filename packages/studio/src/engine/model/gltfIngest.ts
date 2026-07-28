// engine/model/gltfIngest.ts — turn a glTF/GLB file into Parts.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import type { Part } from './part';
import { MICRONS_PER_METER } from './part';
import { resolveMaterialKey } from './materials';

/* How to interpret the file when it deviates from the glTF spec defaults. */
export interface IngestOptions {
  /* 
    glTF usually exports in meters, so MICRONS_PER_METER conversion is correct. However, if a model was actually in
    millimeters but the CAD tool labeled it as meters, everything is too large. Therefore, override allows you to change the scale
    to build the matrix that rescales every vertex.
   */
  micronsPerFileUnit?: number;
  /** glTF is Y-up by spec; CAD exports are often Z-up. 'z' applies the fix. */
  upAxis?: 'y' | 'z';
}

// 
const GLTF_DEFAULTS: Required<IngestOptions> = {
  micronsPerFileUnit: MICRONS_PER_METER,
  upAxis: 'y',
};

/** Parse a .glb/.gltf buffer into Parts (µm, viewer frame, transformations applied). */
export async function ingestGltf(fileBuffer: ArrayBuffer, options: IngestOptions = {},): Promise<Part[]> { // fileBuffer = bytes of gltf model
  const { micronsPerFileUnit, upAxis } = { ...GLTF_DEFAULTS, ...options }; // ... copies all key-value pairs; any overlapping keys in options OVERRIDE GLTF_DEFAULTS
  const gltf = await parseGltf(fileBuffer);

  // Normalization: file units → µm, then Z-up → Y-up if needed. Applied to every vertex once.
  const toCanonical = buildCanonicalTransform(micronsPerFileUnit, upAxis);

  gltf.scene.updateMatrixWorld(true); // Applying the transform.

  const parts: Part[] = [];

  gltf.scene.traverse((node: THREE.Object3D) => {
    if (!(node as THREE.Mesh).isMesh) return;
    const mesh = node as THREE.Mesh;

    // Bake the node's placement into the vertices so a Part is self-contained:
    // downstream code (framing, section, explode) never re-applies transforms.
    const geometry = mesh.geometry.clone();
    geometry.applyMatrix4(mesh.matrixWorld);
    geometry.applyMatrix4(toCanonical);
    geometry.computeBoundingBox();

    const name = bestNameFor(mesh, parts.length);
    parts.push({
      id: `${name}#${parts.length}`,
      name,
      materialKey: resolveMaterialKey(name, cadMaterialNameOf(mesh)),
      geometry,
      transform: mesh.matrixWorld.clone(),
      boundingBox: geometry.boundingBox!.clone(),
    });
  });

  if (parts.length === 0) {
    throw new Error('The file contains no meshes — nothing to display.');
  }
  return parts;
}

function parseGltf(buffer: ArrayBuffer): Promise<{ scene: THREE.Group }> {
  const loader = new GLTFLoader();
  return new Promise((resolve, reject) =>
    loader.parse(buffer, '', resolve, (err: unknown) =>
      reject(err instanceof Error ? err : new Error('Could not parse the glTF file.')),
    ),
  );
}

function buildCanonicalTransform(
  micronsPerFileUnit: number,
  upAxis: 'y' | 'z',
): THREE.Matrix4 {
  const scaleToMicrons = new THREE.Matrix4().makeScale(
    micronsPerFileUnit,
    micronsPerFileUnit,
    micronsPerFileUnit,
  );
  if (upAxis === 'y') return scaleToMicrons; // well-formed glTF: no rotation
  // Z-up file: one −90° rotation about X maps its Z (up) onto our Y (up).
  const zUpToYUp = new THREE.Matrix4().makeRotationX(-Math.PI / 2);
  return zUpToYUp.multiply(scaleToMicrons);
}

/* A mesh's own name, else its parent group's (CAD exporters vary), else a stand-in. */
function bestNameFor(mesh: THREE.Mesh, ordinal: number): string {
  if (mesh.name) return mesh.name;
  if (mesh.parent?.name) return mesh.parent.name;
  return `part-${ordinal + 1}`;
}

/* Gets the material name from a mesh if the CAD tool embedded one. */
function cadMaterialNameOf(mesh: THREE.Mesh): string | null {
  let material;
  // If there are multiple materials for the mesh, use the first one.
  if (Array.isArray(mesh.material)) {
    material = mesh.material[0];
  } else {
    material = mesh.material;
  }
  // Checks mesh.userData.material (extra metadata) for any material info.
  if (material && material.name) {
    return material.name;
  } else if (mesh.userData && mesh.userData.material) {
    return mesh.userData.material as string;
  } else {
    return null;
  }
}
