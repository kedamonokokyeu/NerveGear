// Vite build for NerveGear Studio.
//
// Three.js is pinned at 0.160.0 and VENDORED: both `three` and `three/addons/`
// resolve to ./vendor/three so the pin holds and the app works offline. No CDN.
// `python run_studio.py` serves the dist/ build on :8777; `npm run dev` (:5173)
// proxies the backend at NERVEGEAR_BACKEND (default localhost:8200).
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

const BACKEND = process.env.NERVEGEAR_BACKEND || 'http://localhost:8200';
const r = (p: string) => fileURLToPath(new URL(p, import.meta.url));

export default defineConfig({
  base: './',
  plugins: [react()],
  resolve: {
    alias: [
      { find: /^three\/addons\/(.*)/, replacement: r('./vendor/three/addons/') + '$1' },
      { find: /^three$/, replacement: r('./vendor/three/three.module.js') },
    ],
  },
  server: {
    port: 5173,
    proxy: {
      '/api': BACKEND,
      '/rag': BACKEND,
      '/cad': BACKEND,
      '/health': BACKEND,
    },
  },
  build: {
    outDir: 'dist',
    target: 'es2020',
    chunkSizeWarningLimit: 1400,
  },
});
