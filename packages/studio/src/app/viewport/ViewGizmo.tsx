// Standard-views gizmo — the intuitive navigation aid. Click a named view to
// snap the orthographic camera to it around the current pivot; the reframe
// button (also F) fits the whole model. Delivers the ViewCube's *function*
// (instant standard views) with zero drag. A live rotating cube can layer on
// top later as polish.

export type ViewName = 'iso' | 'top' | 'bottom' | 'front' | 'back' | 'left' | 'right';

interface ViewGizmoProps {
  onView: (name: ViewName) => void;
  onReframe: () => void;
}

// Cross layout: Top over the Front/Left/Iso/Right row, Bottom + Back beneath.
const GRID: (ViewName | null)[] = [
  null, 'top', null,
  'left', 'front', 'right',
  null, 'bottom', null,
];

const LABEL: Record<ViewName, string> = {
  iso: 'Iso',
  top: 'Top',
  bottom: 'Bot',
  front: 'Front',
  back: 'Back',
  left: 'Left',
  right: 'Right',
};

export function ViewGizmo({ onView, onReframe }: ViewGizmoProps) {
  return (
    <div className="viewgizmo" role="group" aria-label="Standard views">
      <div className="viewgizmo__grid">
        {GRID.map((v, i) =>
          v ? (
            <button
              key={v}
              className="viewgizmo__btn"
              title={`${LABEL[v]} view`}
              onClick={() => onView(v)}
            >
              {LABEL[v]}
            </button>
          ) : (
            <span key={`gap-${i}`} className="viewgizmo__gap" />
          ),
        )}
      </div>
      <div className="viewgizmo__row">
        <button className="viewgizmo__btn" title="Isometric view" onClick={() => onView('iso')}>
          Iso
        </button>
        <button className="viewgizmo__btn" title="Reframe model (F)" onClick={onReframe}>
          ⤢ Fit
        </button>
      </div>
    </div>
  );
}
