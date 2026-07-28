import { useAppState } from '../state/store';

/** Bottom status bar (26px). Facts only: model, view, import errors, fps. */
export function StatusBar() {
  const fps = useAppState((s) => s.fps);
  const viewMode = useAppState((s) => s.viewMode);
  const modelName = useAppState((s) => s.modelName);
  const partCount = useAppState((s) => s.parts.length);
  const importError = useAppState((s) => s.importError);
  const backendStatus = useAppState((s) => s.backendStatus);
  const engineName = useAppState((s) => s.engineName);

  return (
    <div className="statusbar">
      <span className="statusbar__field">
        Model: <b>{modelName ?? '—'}</b>
        {partCount > 0 ? <> ({partCount} parts)</> : null}
      </span>
      <span className="statusbar__field">
        View: <b>{viewMode === 'orbit' ? 'structure' : viewMode}</b>
      </span>
      <span className="statusbar__field">
        Backend: <b>{backendStatus ?? '—'}</b>
      </span>
      <span className="statusbar__field">
        Engine: <b>{engineName ?? '—'}</b>
      </span>
      {importError ? (
        <span className="statusbar__field statusbar__error" title={importError}>
          Import failed: {importError}
        </span>
      ) : null}
      <span className="statusbar__spacer" />
      <span className="statusbar__field statusbar__fps">
        {fps === null ? '—' : Math.round(fps)} fps
      </span>
    </div>
  );
}
