/** Right-top pane — the process / checkpoint history (Step 0 → N.M).
 *  Branch-capable tree data model, linear list UI — arrives in Phase 3. */
export function ProcessTreePane() {
  return (
    <div className="empty-state">
      <div className="empty-state__title">No process steps</div>
      <div className="empty-state__hint">
        Etch / deposition checkpoints appear here as a step history (Phase 3).
      </div>
    </div>
  );
}
