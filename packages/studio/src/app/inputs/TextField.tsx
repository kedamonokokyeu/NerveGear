export interface TextFieldProps {
  label: string;
  value: string;
  placeholder?: string;
  disabled?: boolean;
  onChange: (v: string) => void;
}

export function TextField({ label, value, placeholder, disabled, onChange }: TextFieldProps) {
  return (
    <label className="field-row">
      <span className="field-label">{label}</span>
      <span className="field-box">
        <input
          className="field-input field-input--text"
          type="text"
          value={value}
          placeholder={placeholder}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
        />
      </span>
    </label>
  );
}
