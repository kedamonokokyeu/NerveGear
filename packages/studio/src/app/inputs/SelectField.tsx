export interface SelectOption {
  value: string;
  label: string;
}

export interface SelectFieldProps {
  label: string;
  value: string;
  options: SelectOption[];
  disabled?: boolean;
  onChange: (v: string) => void;
}

/** Enterprise-styled native select. */
export function SelectField({ label, value, options, disabled, onChange }: SelectFieldProps) {
  return (
    <label className="field-row">
      <span className="field-label">{label}</span>
      <span className="field-box">
        <select
          className="field-input field-input--select"
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
        >
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </span>
    </label>
  );
}
