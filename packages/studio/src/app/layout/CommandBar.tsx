/*
React component — the top command bar (44px). Phase 1: Import CAD is live
(picker + the same pipeline drag-drop uses), the Orbit/Section/Explode
switch drives the real tools, and each tool's typed controls appear inline
next to the switch while that tool is active. Export ships in a later phase.
*/

import { useRef } from 'react';
import { useTheme } from '../theme/ThemeProvider';
import { setState, useAppState, type ViewMode } from '../state/store';
import { importModelFile } from '../state/modelSession';
import { SUPPORTED_EXTENSIONS } from '../../engine/model/loadModel';
import { NumberField, SelectField } from '../inputs';
import type { SectionAxis } from '../../engine/model/section';

const VIEW_MODES: { id: ViewMode; label: string }[] = [
  { id: 'orbit', label: 'Orbit' },
  { id: 'section', label: 'Section' },
  { id: 'explode', label: 'Explode' },
];

const AXIS_OPTIONS: { value: SectionAxis; label: string }[] = [
  { value: 'y', label: 'Y (depth)' },
  { value: 'x', label: 'X (flow)' },
  { value: 'z', label: 'Z (transverse)' },
];

const FILE_ACCEPT = SUPPORTED_EXTENSIONS.map((ext) => `.${ext}`).join(',');

export function CommandBar() {
  const { theme, toggleTheme } = useTheme();
  const viewMode = useAppState((s) => s.viewMode);
  const importPhase = useAppState((s) => s.importPhase);
  const sectionAxis = useAppState((s) => s.sectionAxis);
  const sectionDepthMicrons = useAppState((s) => s.sectionDepthMicrons);
  const explodeGapMicrons = useAppState((s) => s.explodeGapMicrons);
  const legendVisible = useAppState((s) => s.legendVisible);
  const filePickerRef = useRef<HTMLInputElement>(null);

  async function onFilePicked(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    // Errors land in store.importError (surfaced by the status bar and the
    // Structure pane); the picker's job ends at handing the file over.
    await importModelFile(file).catch(() => {});
  }

  return (
    <div className="commandbar">
      <span className="commandbar__brand">NerveGear Studio</span>
      <span className="commandbar__divider" />
      <input
        ref={filePickerRef}
        type="file"
        accept={FILE_ACCEPT}
        style={{ display: 'none' }}
        onChange={(e) => {
          void onFilePicked(e.target.files);
          e.target.value = ''; // allow re-importing the same file
        }}
      />
      <button
        className="cmd-btn"
        disabled={importPhase === 'loading'}
        onClick={() => filePickerRef.current?.click()}
        title={`Import a CAD model (${FILE_ACCEPT})`}
      >
        {importPhase === 'loading' ? 'Importing…' : 'Import CAD'}
      </button>
      <button className="cmd-btn" disabled title="Phase 4">
        Export
      </button>
      <span className="commandbar__divider" />
      <div className="segmented" role="group" aria-label="View mode">
        {VIEW_MODES.map((m) => (
          <button
            key={m.id}
            className={'segmented__btn' + (viewMode === m.id ? ' segmented__btn--active' : '')}
            onClick={() => setState({ viewMode: m.id })}
          >
            {m.label}
          </button>
        ))}
      </div>

      {viewMode === 'section' && (
        <div className="commandbar__tool-controls">
          <SelectField
            label="Axis"
            value={sectionAxis}
            options={AXIS_OPTIONS}
            onChange={(axis) => setState({ sectionAxis: axis as SectionAxis })}
          />
          <NumberField
            label="Depth"
            value={sectionDepthMicrons}
            unit="um"
            min={0}
            step={50}
            decimals={0}
            onChange={(depth) => setState({ sectionDepthMicrons: depth })}
          />
        </div>
      )}

      {viewMode === 'explode' && (
        <div className="commandbar__tool-controls">
          <NumberField
            label="Gap"
            value={explodeGapMicrons}
            unit="um"
            min={0}
            max={20000}
            step={100}
            decimals={0}
            onChange={(gap) => setState({ explodeGapMicrons: gap })}
          />
        </div>
      )}

      <span className="commandbar__spacer" />
      <button
        className="cmd-btn"
        onClick={() => setState({ legendVisible: !legendVisible })}
        title="Show or hide the material color legend"
      >
        {legendVisible ? 'Hide legend' : 'Legend'}
      </button>
      <button className="cmd-btn" onClick={toggleTheme} title="Toggle theme">
        {theme === 'dark' ? 'Light mode' : 'Dark mode'}
      </button>
    </div>
  );
}
