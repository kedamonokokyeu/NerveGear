import { useEffect, useRef, useState } from 'react';
import { clamp, formatValue, parseNumeric, unitLabel } from '../format/units';

export interface NumberFieldProps {
  label: string;
  value: number;
  unit?: string;
  min?: number;
  max?: number;
  step?: number;
  decimals?: number;
  disabled?: boolean;
  onChange: (v: number) => void;
}

/**
 * Typed numeric field — THE way numbers are entered in NerveGear Studio.
 * Commits on Enter/blur; rejects non-numeric input; clamps to [min, max].
 * There is no slider anywhere in this codebase, by decree.
 */
export function NumberField({
  label,
  value,
  unit,
  min,
  max,
  step,
  decimals,
  disabled,
  onChange,
}: NumberFieldProps) {
  const [draft, setDraft] = useState(() => formatValue(value, unit, decimals));
  const [invalid, setInvalid] = useState(false);
  const focused = useRef(false);

  // Reflect external value changes when not mid-edit.
  useEffect(() => {
    if (!focused.current) {
      setDraft(formatValue(value, unit, decimals));
      setInvalid(false);
    }
  }, [value, unit, decimals]);

  function commit() {
    const parsed = parseNumeric(draft);
    if (parsed === null) {
      // Reject: restore last good value.
      setDraft(formatValue(value, unit, decimals));
      setInvalid(false);
      return;
    }
    const next = clamp(parsed, min, max);
    setDraft(formatValue(next, unit, decimals));
    setInvalid(false);
    if (next !== value) onChange(next);
  }

  function nudge(dir: 1 | -1) {
    const s = step ?? 1;
    const base = parseNumeric(draft) ?? value;
    const next = clamp(base + dir * s, min, max);
    setDraft(formatValue(next, unit, decimals));
    if (next !== value) onChange(next);
  }

  return (
    <label className="field-row">
      <span className="field-label">{label}</span>
      <span className={'field-box' + (invalid ? ' field-box--invalid' : '')}>
        <input
          className="field-input field-input--num"
          type="text"
          inputMode="decimal"
          value={draft}
          disabled={disabled}
          onFocus={() => {
            focused.current = true;
          }}
          onChange={(e) => {
            setDraft(e.target.value);
            setInvalid(parseNumeric(e.target.value) === null);
          }}
          onBlur={() => {
            focused.current = false;
            commit();
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
            else if (e.key === 'Escape') {
              setDraft(formatValue(value, unit, decimals));
              setInvalid(false);
              (e.target as HTMLInputElement).blur();
            } else if (e.key === 'ArrowUp') {
              e.preventDefault();
              nudge(1);
            } else if (e.key === 'ArrowDown') {
              e.preventDefault();
              nudge(-1);
            }
          }}
        />
        {unit ? <span className="field-unit">{unitLabel(unit)}</span> : null}
      </span>
    </label>
  );
}
