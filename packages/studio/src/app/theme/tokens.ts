// NerveGear Studio design tokens — the single source of truth for palette, type,
// and spacing. Values mirror theme.css exactly (see PHASE0_BUILD_PROMPT.md §4.5).

export type ThemeName = 'dark' | 'light';

export const THEME_STORAGE_KEY = 'nervegear.theme';

export const palette = {
  dark: {
    bgApp: '#0e1116',
    bgPanel: '#151a21',
    bgElev: '#1b222b',
    bgInput: '#10151b',
    border: '#262e39',
    borderStrong: '#323c49',
    text: '#e6ebf2',
    text2: '#9aa7b8',
    textMuted: '#6b7787',
    accent: '#3b82f6',
    accentHover: '#2f6fd6',
    selection: '#38bdf8',
    danger: '#ef4444',
    ok: '#22c55e',
  },
  light: {
    bgApp: '#f5f7fa',
    bgPanel: '#ffffff',
    bgElev: '#eef1f5',
    bgInput: '#ffffff',
    border: '#d4dae2',
    borderStrong: '#b8c1cd',
    text: '#12181f',
    text2: '#4a5666',
    textMuted: '#7a8494',
    accent: '#2563eb',
    accentHover: '#1d4ed8',
    selection: '#0ea5e9',
    danger: '#dc2626',
    ok: '#16a34a',
  },
} as const;

// Material legend palette (shared across themes; used from Phase 1 on).
// Color = material, everywhere, always.
export const materials = {
  carrierCu: '#c77f3f',
  silicon: '#7d93ad',
  coolant: '#1fb6e0',
  lid: '#9fb6c9',
  dielectricSiO2: '#5b6b7f',
  solderTIM: '#d9a441',
} as const;

export const type = {
  sans: 'ui-sans-serif, system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  mono: 'ui-monospace, "Cascadia Code", "JetBrains Mono", Consolas, monospace',
  baseSize: 12,
  lineHeight: 1.4,
  paneHeaderSize: 11,
  paneHeaderTracking: '0.04em',
} as const;

export const spacing = [4, 8, 12, 16] as const;

export const layout = {
  commandBarH: 44,
  statusBarH: 26,
  left: { default: 260, min: 200, max: 420 },
  right: { default: 340, min: 280, max: 520 },
  processTreeFraction: 0.45,
} as const;
