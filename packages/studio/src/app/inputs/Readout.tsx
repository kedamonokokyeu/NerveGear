import { formatValue, unitLabel } from '../format/units';

export interface ReadoutProps {
  label: string;
  value: number | string;
  unit?: string;
  decimals?: number;
}

/** Read-only labeled value (monospace) for derived / displayed numbers. */
export function Readout({ label, value, unit, decimals }: ReadoutProps) {
  const text =
    typeof value === 'number' ? formatValue(value, unit, decimals) : value;
  return (
    <div className="field-row field-row--readout">
      <span className="field-label">{label}</span>
      <span className="readout-value">
        {text}
        {unit ? <span className="field-unit">{unitLabel(unit)}</span> : null}
      </span>
    </div>
  );
}
