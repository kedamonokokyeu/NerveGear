// engine/model/stepConversion.ts — send a STEP file to the backend, get GLB back.

// Kept relative so the Vite dev proxy and any same-origin deployment both work.
const STEP_CONVERSION_URL = '/cad/convert/step-to-glb';

// The backend states what one unit of the returned GLB means, in µm. Its
// converter (OpenCASCADE) normalises every STEP to glTF-spec metres, so this
// is 1e6 today — but the client trusts the header, not that implementation
// detail, so the backend is free to change converters.
const SOURCE_UNIT_HEADER = 'x-nervegear-microns-per-unit';

export interface StepConversionResult {
  glbBuffer: ArrayBuffer;
  /** µm per unit in the returned GLB — feed straight into ingestGltf(). */
  micronsPerFileUnit: number;
}

/** Upload a .step/.stp file; resolve to GLB bytes ready for ingestGltf(). */
export async function convertStepToGlb(file: File): Promise<StepConversionResult> {
  const form = new FormData();
  form.append('file', file, file.name);

  let response: Response;
  try {
    response = await fetch(STEP_CONVERSION_URL, { method: 'POST', body: form });
  } catch {
    throw new Error(
      'STEP conversion needs the NerveGear backend running (it is not reachable). ' +
        'GLB/glTF files load without it.',
    );
  }

  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }

  // Missing header (old backend build): assume glTF-spec metres.
  const micronsPerFileUnit =
    Number(response.headers.get(SOURCE_UNIT_HEADER)) || 1_000_000;
  return { glbBuffer: await response.arrayBuffer(), micronsPerFileUnit };
}

/** Pull the backend's human-readable error out of a failed response. */
async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === 'string') return body.detail;
  } catch {
    // Not JSON — fall through to the generic message.
  }
  return `STEP conversion failed (HTTP ${response.status}).`;
}
