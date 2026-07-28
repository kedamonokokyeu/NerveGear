// engine/model/loadModel.ts — the one front door for importing a CAD file.
/* From the picker or through drag-drop, software determines the format from the extension, routes it down
the correct path (e.g., STEP has to go down the conversion into glTF, etc.). Returns a LoadedModel with named Parts
in micrometer orientations for the mesh.
*/
//   .glb / .gltf  -> parsed right here in the browser 
//   .step / .stp  -> uploaded to the backend, converted to GLB, then parsed

import * as THREE from 'three';
import type { LoadedModel, Part } from './part';
import { ingestGltf, type IngestOptions } from './gltfIngest';
import { convertStepToGlb } from './stepConversion';

export const SUPPORTED_EXTENSIONS = ['glb', 'gltf', 'step', 'stp'] as const; // Items within the list are immutable (const).

function extensionOf(fileName: string): string {
  return fileName.slice(fileName.lastIndexOf('.') + 1).toLowerCase();
}

export function isSupportedCadFile(fileName: string): boolean {
  return (SUPPORTED_EXTENSIONS as readonly string[]).includes(extensionOf(fileName)); // extensionOf extracts the file type.
}

// Imports CAD file into Part contract. Overrides are if there are incorrect values for the part. Lets you manually apply replace values for a part through UI.
export async function loadModelFromFile(file: File, overrides: IngestOptions = {},): Promise<LoadedModel> {
  const extension = extensionOf(file.name);

  let parts: Part[];
  // Routing for file extension type.
  switch (extension) {
    case 'glb':
    case 'gltf': {
      parts = await ingestGltf(await file.arrayBuffer(), overrides);
      break;
    }
    case 'step':
    case 'stp': {
      const { glbBuffer, micronsPerFileUnit } = await convertStepToGlb(file);
      // The user's unit override still wins over the backend's detection.
      parts = await ingestGltf(glbBuffer, { micronsPerFileUnit, ...overrides });
      break;
    }
    default:
      throw new Error(
        `Unsupported file type ".${extension}" — supported: ${SUPPORTED_EXTENSIONS.join(', ')}.`,
      );
  }

  return { fileName: file.name, parts, boundingBox: unionOfBounds(parts) };
}

function unionOfBounds(parts: Part[]): THREE.Box3 {
  /* Each part has its own bounding box; here we merge all parts into one large bounding box (for entire CAD export). */
  const union = new THREE.Box3();
  for (const part of parts) union.union(part.boundingBox);
  return union;
}
