/*
React component — the "did we read your file right?" dialog. It opens only
when an imported model's size is implausible for a chip (a sign the file's
declared units, or its up-axis, are wrong — both are common in CAD exports).
The user picks what the file's numbers actually mean and we re-import the
same file with those overrides.
*/

import { useState } from 'react';
import { useAppState, setState } from '../state/store';
import { reimportWithOverrides } from '../state/modelSession';
import { SelectField } from '../inputs';
import {
  MICRONS_PER_METER,
  MICRONS_PER_MILLIMETER,
  formatLengthMicrons,
} from '../../engine/model/part';

const UNIT_OPTIONS = [
  { value: String(MICRONS_PER_METER), label: 'metres (glTF spec)' },
  { value: String(MICRONS_PER_MILLIMETER), label: 'millimetres' },
  { value: '1', label: 'micrometres' },
];

const UP_AXIS_OPTIONS = [
  { value: 'y', label: 'Y up (glTF spec)' },
  { value: 'z', label: 'Z up (common CAD)' },
];

export function ImportReviewDialog() {
  const open = useAppState((s) => s.importNeedsReview);
  const parts = useAppState((s) => s.parts);
  const [micronsPerUnit, setMicronsPerUnit] = useState(String(MICRONS_PER_MILLIMETER));
  const [upAxis, setUpAxis] = useState<'y' | 'z'>('y');
  if (!open) return null;

  const largestDimensionMicrons = Math.max(
    1,
    ...parts.flatMap((p) => [p.sizeMicrons.x, p.sizeMicrons.y, p.sizeMicrons.z]),
  );

  const keepAsIs = () => setState({ importNeedsReview: false });
  const applyOverrides = () =>
    void reimportWithOverrides({
      micronsPerFileUnit: Number(micronsPerUnit),
      upAxis,
    }).catch(() => {}); // failure surfaces through store.importError

  return (
    <div className="modal-backdrop">
      <div className="modal" role="dialog" aria-label="Review import units">
        <div className="modal__title">Check this file's units</div>
        <div className="modal__body">
          Read as declared, the model is about {formatLengthMicrons(largestDimensionMicrons)}{' '}
          across — unusual for a chip or package. If that's wrong, say what the
          file's numbers really mean and it will be re-imported.
        </div>
        <SelectField
          label="File units"
          value={micronsPerUnit}
          options={UNIT_OPTIONS}
          onChange={setMicronsPerUnit}
        />
        <SelectField
          label="Up axis"
          value={upAxis}
          options={UP_AXIS_OPTIONS}
          onChange={(v) => setUpAxis(v as 'y' | 'z')}
        />
        <div className="modal__actions">
          <button className="cmd-btn" onClick={keepAsIs}>
            Keep as is
          </button>
          <button className="cmd-btn" onClick={applyOverrides}>
            Re-import
          </button>
        </div>
      </div>
    </div>
  );
}
