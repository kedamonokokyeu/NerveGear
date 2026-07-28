// unit + number processing --- displayed numbers on frontend go through this process
// specifies how each unit looks (micrometers, Celsius, etc.)

// any variable of type UnitKind can only be the following strings:
export type UnitKind =
  | 'um' | 'mm' | 'C' | 'K' | 'kPa' | 'Pa' | 'mL/min' | 'W' | 'W/cm2' | 'none';

export interface UnitSpec {
  /* Shape of an object that describes how a unit should appear (label, and decimals). */
  label: string;
  decimals: number;
}

export const UNITS: Record<UnitKind, UnitSpec> = {
  um: { label: 'µm', decimals: 1 },
  mm: { label: 'mm', decimals: 3 },
  C: { label: '°C', decimals: 1 },
  K: { label: 'K', decimals: 1 },
  kPa: { label: 'kPa', decimals: 2 },
  Pa: { label: 'Pa', decimals: 0 },
  'mL/min': { label: 'mL/min', decimals: 1 },
  W: { label: 'W', decimals: 1 },
  'W/cm2': { label: 'W/cm²', decimals: 1 },
  none: { label: '', decimals: 2 },
};

// Restricts number to a range, e.g., clamp(5, 0, 3) gives 3.
export function clamp(v: number, min?: number, max?: number): number {
  if (min !== undefined && v < min) return min;
  if (max !== undefined && v > max) return max;
  return v;
}

/** Format value for display. `decimals` overrides the unit default. */
export function formatValue(v: number, unit?: string, decimals?: number): string {
  if (!Number.isFinite(v)) return '—'; // checks if v is real & finite

  let spec;
  if (unit && unit in UNITS) {
    spec = UNITS[unit as UnitKind]
  } else {
    spec = undefined;
  }

  let d;
  if (decimals !== undefined) {
    d = decimals;
  } else if (spec && spec.decimals !== undefined) {
    d = spec.decimals;
  } else {
    d = 2;
  }
  return v.toFixed(d); // fixes number of decimal points
}

/** Resolve the display suffix for a unit key (falls back to the raw string). */
export function unitLabel(unit?: string): string {
  if (!unit) return '';
  if (unit in UNITS) {
    return UNITS[unit as UnitKind].label;
  } else {
    return unit;
  }
}

/** Strict numeric parse: trims, rejects empty/partial input. */
export function parseNumeric(raw: string): number | null {
  const s = raw.trim();
  if (s === '') return null;
  const v = Number(s);
  if (Number.isFinite(v)) {
    return v;
  } else {
    return null;
  }
}
