// engine/model/part.ts
/* 
Since we want part analysis later and broken down sections of a CAD object, this is
where the structure of a 'Part' is defined. 

Named solids with labels in ingested CA file will be kept track of.
Each individual named solid becomes a 'Part' --- follows a record convention that the app agrees on.
3D scene, structure pane, and properties panel read parts --- raw file remains untouched.

Units contract:
  - each Part's geometry is in micrometers, in the viewer frame (X = coolant flow, Y = die stack growing up, Z = transverse)
  - conversion from file's declared unit to micrometers occurs in the loader (loadModel.ts)
  - rendering scales whole scene by RENDER_UNITS_PER_MICRON (camera math remains consistent, 1 render unit = 1 mm)
*/


import type * as THREE from 'three';

export interface Part {
  id: string; // Unique ID within loaded model (name + ordinal).
  name: string; // CAD file name.
  materialKey: string; // Semantic material key for coloring.
  geometry: THREE.BufferGeometry; // Triangle geometry from the mesh in micrometers.
  transform: THREE.Matrix4; // 4 x 4 Transformation matrix -- tells where the part is in 3D space.
  boundingBox: THREE.Box3; // Rectangular box, fully contains 3D object. 
}

/* CAD export after ingestion */
export interface LoadedModel {
  fileName: string;
  parts: Part[];
  boundingBox: THREE.Box3; // Example: const box = new THREE.Box3().setFromObject(mesh)
}

// glTF / glB --> 3D file format: stores models into meshes (also STEP can be converted into this, less accurate geometry than STEP models, though.)
// Vertex coordinates stored in meters --> conversion to micrometer constants for future reference here:
export const MICRONS_PER_METER = 1_000_000;
export const MICRONS_PER_MILLIMETER = 1_000;

// Geometry is in micrometers. However, CAD data lives in micrometers --- too much heavy processing for camera view.
// 1 micrometer in the model is 0.001 render units, converting micrometers --> millimeters for display.
export const RENDER_UNITS_PER_MICRON = 0.001;

/** Display rule for units in UI --> if length below 10,000 micrometers, show in micrometers. Otherwise, millimeters. */
export const DISPLAY_MM_CUTOFF_MICRONS = 10_000;

export function formatLengthMicrons(valueMicrons: number): string {
  if (Math.abs(valueMicrons) >= DISPLAY_MM_CUTOFF_MICRONS) {
    return `${(valueMicrons / MICRONS_PER_MILLIMETER).toFixed(2)} mm`; // 2 decimal point accuracy
  }
  return `${valueMicrons.toFixed(1)} µm`;
}
