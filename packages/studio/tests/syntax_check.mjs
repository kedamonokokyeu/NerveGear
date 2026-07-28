// Zero-dependency smoke: every plain-JS engine module must parse as valid ESM.
// (TS/TSX is covered by `tsc --noEmit`; e2e lives in tests/phase1.spec.js.)
import { readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';

let bad = 0;
for (const f of readdirSync('src/engine').filter((x) => x.endsWith('.js'))) {
  try {
    execFileSync(process.execPath, ['--check', `src/engine/${f}`], { stdio: 'pipe' });
    console.log(`  ok  src/engine/${f}`);
  } catch (e) {
    bad++;
    console.error(`FAIL  src/engine/${f}\n${e.stderr}`);
  }
}
process.exit(bad ? 1 : 0);
