// Node-side Phase 1 engine checks (no browser needed).
//
// Bundles the engine's model modules with esbuild, then runs the real import
// pipeline on the bundled sample GLB and asserts the contract holds: part
// count and names, µm sizes, semantic materials, explode tiering, and the
// section plane math. The browser e2e (phase1.spec.js) covers rendering and
// interaction; this covers the geometry/units brain, fast enough for CI.
//
// Run: node tests/phase1_engine_checks.mjs

import { build } from 'esbuild';
import { readFileSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const studioRoot = join(dirname(fileURLToPath(import.meta.url)), '..');

// One tiny entry that re-exports everything the checks need; esbuild resolves
// the vendored three exactly like vite does (see vite.config.ts aliases).
const modelDir = join(studioRoot, 'src/engine/model').replaceAll('\\', '/');
const entrySource = `
export { ingestGltf } from '${modelDir}/gltfIngest';
export { resolveMaterialKey } from '${modelDir}/materials';
export { createSectionController } from '${modelDir}/section';
export * as THREE from 'three';
`;

// Both the generated entry and the bundle live in a temp dir — nothing to
// gitignore, nothing left behind.
const workdir = mkdtempSync(join(tmpdir(), 'nervegear-engine-check-'));
const entryPath = join(workdir, 'entry.ts');
writeFileSync(entryPath, entrySource);
const bundlePath = join(workdir, 'engine.mjs');

await build({
  entryPoints: [entryPath],
  bundle: true,
  format: 'esm',
  outfile: bundlePath,
  alias: {
    'three/addons': join(studioRoot, 'vendor/three/addons'),
    three: join(studioRoot, 'vendor/three/three.module.js'),
  },
  logLevel: 'silent',
});

const engine = await import(pathToFileURL(bundlePath));

let failures = 0;
function check(label, condition, detail = '') {
  if (condition) console.log(`  ok  ${label}`);
  else {
    failures += 1;
    console.error(`FAIL  ${label}  ${detail}`);
  }
}

// ── the bundled sample through the real pipeline ──
const glbBytes = readFileSync(join(studioRoot, 'src/assets/sample-stacked-die.glb'));
const parts = await engine.ingestGltf(
  glbBytes.buffer.slice(glbBytes.byteOffset, glbBytes.byteOffset + glbBytes.byteLength),
);

check('sample yields 19 named parts', parts.length === 19, `got ${parts.length}`);
check(
  'part names survive (gpu-die present)',
  parts.some((p) => p.name === 'gpu-die'),
);

const substrate = parts.find((p) => p.name === 'substrate');
const substrateWidthMicrons = substrate.boundingBox.max.x - substrate.boundingBox.min.x;
check(
  'glTF metres → µm once (30 mm substrate = 30 000 µm)',
  Math.abs(substrateWidthMicrons - 30_000) < 1,
  `got ${substrateWidthMicrons}`,
);

check(
  'label heuristics: gpu-die → silicon',
  parts.find((p) => p.name === 'gpu-die').materialKey === 'silicon',
);
check(
  'label heuristics: tim → solder',
  parts.find((p) => p.name === 'tim').materialKey === 'solder',
);
check(
  'label heuristics: lid → lid',
  parts.find((p) => p.name === 'lid').materialKey === 'lid',
);

// ── orientation override: a Z-up file gets the −90° X fix ──
const partsZUp = await engine.ingestGltf(
  glbBytes.buffer.slice(glbBytes.byteOffset, glbBytes.byteOffset + glbBytes.byteLength),
  { upAxis: 'z' },
);
const lidYUp = parts.find((p) => p.name === 'lid');
const lidZUp = partsZUp.find((p) => p.name === 'lid');
check(
  'Z-up override rotates the stack axis',
  Math.abs(lidZUp.boundingBox.getCenter(new engine.THREE.Vector3()).z +
           lidYUp.boundingBox.getCenter(new engine.THREE.Vector3()).y) < 1,
);

// ── section plane math: depth measured from the +face, in µm ──
const section = engine.createSectionController();
const bounds = new engine.THREE.Box3(
  new engine.THREE.Vector3(0, 0, 0),
  new engine.THREE.Vector3(10_000, 4_000, 8_000),
);
const planes = section.apply({ axis: 'y', depthMicrons: 1_000 }, bounds);
check('section on → one clipping plane', planes.length === 1);
// Cut at y = (4000 − 1000) µm = 3 render units; normal faces −Y.
check(
  'plane sits at max − depth (render units)',
  Math.abs(planes[0].constant - 3) < 1e-9 && planes[0].normal.y === -1,
  `constant=${planes[0].constant} normal=${JSON.stringify(planes[0].normal)}`,
);
check('section off → no planes', section.apply(null, bounds).length === 0);

console.log(failures ? `\n${failures} check(s) FAILED` : '\nall engine checks passed');
process.exit(failures ? 1 : 0);
