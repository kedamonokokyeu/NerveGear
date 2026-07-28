/*
React component — the material color legend, floating over the viewport.
Color = material everywhere in this app; this small card is the decoder ring.
It only lists materials that actually appear in the loaded model, so it reads
like a caption for what's on screen, not a catalogue.
*/

import { useAppState } from '../state/store';
import { materialInfo } from '../../engine/model/materials';

export function LegendOverlay() {
  const parts = useAppState((s) => s.parts);
  if (parts.length === 0) return null;

  const materialKeysInModel = [...new Set(parts.map((p) => p.materialKey))];

  return (
    <div className="legend" aria-label="Material legend">
      <div className="legend__title">Materials</div>
      {materialKeysInModel.map((key) => {
        const info = materialInfo(key);
        return (
          <div className="legend__row" key={key}>
            <span className="legend__swatch" style={{ background: info.colorHex }} />
            <span className="legend__label">{info.label}</span>
          </div>
        );
      })}
    </div>
  );
}
