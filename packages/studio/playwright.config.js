// Headless e2e for the Studio. The Phase 1 spec (tests/phase1.spec.js) runs
// against the vite build served by run_studio.py on 8777 (dist/ preferred),
// so rebuild first if src changed: npm run build && npx playwright test
//
// tests/smoke.spec.js targeted the superseded pre-React Studio and was
// retired with the Phase 0 rebuild (kept on disk for reference only).
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: 'phase1.spec.js',
  timeout: 45000,
  use: { baseURL: 'http://localhost:8777', viewport: { width: 1440, height: 900 } },
  webServer: {
    command: 'python3 run_studio.py',
    port: 8777,
    reuseExistingServer: true,
    timeout: 15000,
  },
});
