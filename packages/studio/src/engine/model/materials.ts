// engine/model/materials.ts — the semantic material system.
//
// Material is decided in three tiers of trust:
//
//   1. CAD metadata  — the file explicitly says "silicon"        (best)
//   2. Label heuristics — the part's NAME suggests it ("gpu-die") (good)
//   3. Manual override — the user corrects it in Properties       (final say)
//
// The hex values mirror the --mat-* variables in app/theme/theme.css and
// tokens.ts. If you change a color, change it in all three places.

export type MaterialKey =
  | 'silicon'
  | 'copper'
  | 'dielectric'
  | 'solder'
  | 'lid'
  | 'coolant'
  | 'unknown';

export interface MaterialInfo {
  key: MaterialKey;
  /** Legend label a non-specialist can read. */
  label: string;
  /** Shared across themes — color = material, everywhere, always. */
  colorHex: string;
}

export const MATERIAL_LEGEND: MaterialInfo[] = [
  { key: 'silicon', label: 'Silicon (die)', colorHex: '#7d93ad' },
  { key: 'copper', label: 'Copper (carrier / stiffener)', colorHex: '#c77f3f' },
  { key: 'dielectric', label: 'Dielectric (substrate / SiO₂)', colorHex: '#5b6b7f' },
  { key: 'solder', label: 'Solder / TIM', colorHex: '#d9a441' },
  { key: 'lid', label: 'Lid (IHS)', colorHex: '#9fb6c9' },
  { key: 'coolant', label: 'Coolant', colorHex: '#1fb6e0' },
  { key: 'unknown', label: 'Unassigned', colorHex: '#8a8f98' },
];

export function materialInfo(key: string): MaterialInfo {
  return (
    MATERIAL_LEGEND.find((m) => m.key === key) ??
    MATERIAL_LEGEND[MATERIAL_LEGEND.length - 1] // fall back to 'unknown'
  );
}

// ── Tier 1: CAD metadata ────────────────────────────────────────────────────
// glTF materials and node `extras` sometimes carry a material name. We accept
// it when it maps cleanly onto our legend.
export function materialFromCadMetadata(
  metadataName: string | undefined | null,
): MaterialKey | null {
  if (!metadataName) return null;
  return matchMaterialWord(metadataName);
}

// ── Tier 2: label heuristics ────────────────────────────────────────────────
// Real assemblies name their solids ("HBM_stack_3", "cu-stiffener-ring",
// "TIM1"). Each entry maps the words that appear in practice to a legend key.
// Order matters: the first pattern that matches wins, so more specific words
// (e.g. "interposer") come before generic ones (e.g. "si").
const NAME_PATTERNS: { pattern: RegExp; key: MaterialKey }[] = [
  { pattern: /coolant|fluid|water|channel/i, key: 'coolant' },
  { pattern: /lid|ihs|cap\b|cover/i, key: 'lid' },
  { pattern: /solder|tim\b|bump|c4|bga|ball/i, key: 'solder' },
  { pattern: /copper|cu\b|stiffener|mlcc|cap(acitor)?s?\b|trace/i, key: 'copper' },
  { pattern: /substrate|dielectric|sio2|oxide|pcb|organic/i, key: 'dielectric' },
  { pattern: /si\b|silicon|die\b|dies\b|hbm|dram|gpu|cpu|interposer|chip|wafer|tsv/i, key: 'silicon' },
];

export function materialFromLabel(partName: string): MaterialKey | null {
  for (const { pattern, key } of NAME_PATTERNS) {
    if (pattern.test(partName)) return key;
  }
  return null;
}

/** Loose word match used for metadata strings ("Silicon", "Cu", "SolderMask"). */
function matchMaterialWord(text: string): MaterialKey | null {
  return materialFromLabel(text);
}

/**
 * The three-tier resolution, in one place. Manual overrides (tier 3) are
 * applied later by the store when the user edits a part's material — this
 * function only covers what the file itself can tell us.
 */
export function resolveMaterialKey(
  partName: string,
  cadMetadataName?: string | null,
): MaterialKey {
  return (
    materialFromCadMetadata(cadMetadataName) ??
    materialFromLabel(partName) ??
    'unknown'
  );
}
