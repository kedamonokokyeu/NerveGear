// app/state/modelSession.ts — who owns the currently loaded model.
//
//   Import UI (CommandBar, drag-drop) ──► importModelFile()
//                                            │ updates
//                              store (summaries, status) + this session
//                                            │ notifies
//                              ViewportHost ──► rebuilds the 3D scene

import type { LoadedModel } from '../../engine/model/part';
import { loadModelFromFile } from '../../engine/model/loadModel';
import type { IngestOptions } from '../../engine/model/gltfIngest';
import { setState, getState } from './store';

let currentModel: LoadedModel | null = null;

// Kept so the import-review dialog can re-run the SAME file with the user's
// unit / up-axis overrides, without asking them to pick it again.
let lastImportedFile: File | null = null;

// A chip/package model should be somewhere between a bond pad and a server
// tray. Outside this range the declared units are almost certainly wrong,
// and the review dialog opens.
const PLAUSIBLE_MODEL_SIZE_MICRONS = { min: 100, max: 1_000_000 };

type ModelListener = (model: LoadedModel | null) => void;
const modelListeners = new Set<ModelListener>();

/** ViewportHost subscribes here; fires immediately with the current model. */
export function onModelChange(listener: ModelListener): () => void {
  modelListeners.add(listener);
  listener(currentModel);
  return () => modelListeners.delete(listener);
}

export function getCurrentModel(): LoadedModel | null {
  return currentModel;
}

/** The one entry point for both the file picker and drag-drop. */
export async function importModelFile(
  file: File,
  overrides: IngestOptions = {},
): Promise<void> {
  setState({ importPhase: 'loading', importError: null, importNeedsReview: false });
  try {
    const model = await loadModelFromFile(file, overrides);
    lastImportedFile = file;
    adoptModel(model);
    // A hand-picked override means the user has already reviewed the units;
    // only unprompted imports get the plausibility check.
    const userAlreadyReviewed = Object.keys(overrides).length > 0;
    if (!userAlreadyReviewed && !hasPlausibleSize(model)) {
      setState({ importNeedsReview: true });
    }
  } catch (error) {
    setState({
      importPhase: 'error',
      importError: error instanceof Error ? error.message : String(error),
    });
    throw error;
  }
}

/** Re-run the last import with unit / up-axis overrides from the dialog. */
export async function reimportWithOverrides(overrides: IngestOptions): Promise<void> {
  if (!lastImportedFile) return;
  await importModelFile(lastImportedFile, overrides);
}

function hasPlausibleSize(model: LoadedModel): boolean {
  const size = model.boundingBox.max.clone().sub(model.boundingBox.min);
  const largestDimensionMicrons = Math.max(size.x, size.y, size.z);
  return (
    largestDimensionMicrons >= PLAUSIBLE_MODEL_SIZE_MICRONS.min &&
    largestDimensionMicrons <= PLAUSIBLE_MODEL_SIZE_MICRONS.max
  );
}

/** Used by the boot path for the bundled sample (already fetched as a File). */
export function adoptModel(model: LoadedModel): void {
  disposeCurrentGeometry();
  currentModel = model;
  setState({
    importPhase: 'idle',
    importError: null,
    modelName: model.fileName,
    parts: model.parts.map((part) => ({
      id: part.id,
      name: part.name,
      materialKey: part.materialKey,
      sizeMicrons: {
        x: part.boundingBox.max.x - part.boundingBox.min.x,
        y: part.boundingBox.max.y - part.boundingBox.min.y,
        z: part.boundingBox.max.z - part.boundingBox.min.z,
      },
    })),
    selection: null, // the old model's selection means nothing in the new one
  });
  for (const listener of modelListeners) listener(currentModel);
}

/**
 * Tier 3 of the material system: the user's manual override from Properties.
 * Updates the Part record (so re-renders agree) and the store summary; the
 * viewport listens to the store and recolors the live mesh.
 */
export function overridePartMaterial(partId: string, materialKey: string): void {
  const part = currentModel?.parts.find((p) => p.id === partId);
  if (part) part.materialKey = materialKey;
  setState({
    parts: getState().parts.map((summary) =>
      summary.id === partId ? { ...summary, materialKey } : summary,
    ),
  });
}

// The scene (meshes, edge lines) is disposed by the viewport when the model
// changes; the raw part geometries are ours to release.
function disposeCurrentGeometry(): void {
  if (!currentModel) return;
  for (const part of currentModel.parts) part.geometry.dispose();
  currentModel = null;
}
