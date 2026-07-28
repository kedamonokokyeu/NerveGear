// app/import/bootstrapSampleModel.ts — the app is never empty.

import { importModelFile } from '../state/modelSession';
import { getState } from '../state/store';

// Vite bundles the asset and hands back its served URL.
const SAMPLE_MODEL_URL = new URL(
  '../../assets/sample-stacked-die.glb',
  import.meta.url,
).href;

const SAMPLE_FILE_NAME = 'sample-stacked-die.glb';

/** Load the bundled sample unless a model is already in (or entering) the app. */
export async function bootstrapSampleModel(): Promise<void> {
  if (getState().modelName !== null || getState().importPhase === 'loading') return;

  const response = await fetch(SAMPLE_MODEL_URL);
  if (!response.ok) {
    // No sample is not fatal — the empty states explain how to import.
    console.warn('Bundled sample model missing:', response.status);
    return;
  }
  const file = new File([await response.arrayBuffer()], SAMPLE_FILE_NAME, {
    type: 'model/gltf-binary',
  });
  await importModelFile(file);
}
