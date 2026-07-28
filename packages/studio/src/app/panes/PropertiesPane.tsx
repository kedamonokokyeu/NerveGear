/*
React component — the right-bottom pane: the selected part's identity card.
Name and measured size are read-only facts from the file; material is the one
thing the user can override here (tier 3 of the material system — the file's
metadata and the name heuristics are tiers 1 and 2).
*/

import { SelectField, Readout } from '../inputs';
import { useAppState } from '../state/store';
import { overridePartMaterial } from '../state/modelSession';
import { MATERIAL_LEGEND } from '../../engine/model/materials';
import { formatLengthMicrons } from '../../engine/model/part';

const MATERIAL_OPTIONS = MATERIAL_LEGEND.map((m) => ({ value: m.key, label: m.label }));

export function PropertiesPane() {
  const selection = useAppState((s) => s.selection);
  const parts = useAppState((s) => s.parts);
  const selectedPart = parts.find((p) => p.id === selection) ?? null;

  if (!selectedPart) {
    return (
      <div className="empty-state">
        <div className="empty-state__title">Nothing selected</div>
        <div className="empty-state__hint">
          Click a part in the viewport or the Structure list to inspect it.
        </div>
      </div>
    );
  }

  const { x: lengthX, y: heightY, z: widthZ } = selectedPart.sizeMicrons;

  return (
    <div className="props">
      <div className="props__group">
        <div className="props__group-title">Part</div>
        <Readout label="Name" value={selectedPart.name} />
        <SelectField
          label="Material"
          value={selectedPart.materialKey}
          options={MATERIAL_OPTIONS}
          onChange={(key) => overridePartMaterial(selectedPart.id, key)}
        />
      </div>

      <div className="props__group">
        <div className="props__group-title">Dimensions (bounding)</div>
        {/* Viewer frame: X = flow, Y = stack height, Z = transverse. */}
        <Readout label="L (X, flow)" value={formatLengthMicrons(lengthX)} />
        <Readout label="W (Z)" value={formatLengthMicrons(widthZ)} />
        <Readout label="H (Y, stack)" value={formatLengthMicrons(heightY)} />
      </div>

      <div className="props__note">
        Sizes are axis-aligned bounds from the imported mesh. Parametric
        dimensions arrive with the Phase 2 inference sweep.
      </div>
    </div>
  );
}
