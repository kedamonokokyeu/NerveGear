/*
React component — the left pane: every named part in the loaded model, with
its material swatch. Flat list in Phase 1 (the file's own labels, no inferred
hierarchy — that's Phase 2). Selection is bidirectional: clicking a row
highlights the part in 3D, clicking a part in 3D highlights its row.
*/

import { useAppState, setState } from '../state/store';
import { materialInfo } from '../../engine/model/materials';

export function StructurePane() {
  const parts = useAppState((s) => s.parts);
  const selection = useAppState((s) => s.selection);
  const importPhase = useAppState((s) => s.importPhase);

  if (parts.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state__title">
          {importPhase === 'loading' ? 'Importing…' : 'No model loaded'}
        </div>
        <div className="empty-state__hint">
          Import a .glb / .gltf / .step file to list its parts here.
        </div>
      </div>
    );
  }

  return (
    <div className="structure" role="listbox" aria-label="Model parts">
      {parts.map((part) => {
        const isSelected = part.id === selection;
        return (
          <button
            key={part.id}
            role="option"
            aria-selected={isSelected}
            className={'structure__row' + (isSelected ? ' structure__row--selected' : '')}
            // Clicking the selected row again deselects — same as clicking
            // empty space in the viewport.
            onClick={() => setState({ selection: isSelected ? null : part.id })}
            title={part.name}
          >
            <span
              className="structure__swatch"
              style={{ background: materialInfo(part.materialKey).colorHex }}
            />
            <span className="structure__name">{part.name}</span>
          </button>
        );
      })}
    </div>
  );
}
