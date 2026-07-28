// Minimal shared app state. One tiny external store that the panes, viewport,
// command bar, and status bar all read — so selection, tool state, and status
// never need rewiring. Deliberately dependency-free.
//
// What lives here: plain UI-facing values (ids, names, numbers). What does
// NOT live here: Three.js objects and geometry — those stay in the engine,
// coordinated by state/modelSession.ts.

import { useSyncExternalStore } from 'react';
import type { SectionAxis } from '../../engine/model/section';

export type ViewMode = 'orbit' | 'section' | 'explode';

/** The UI's view of one imported part (geometry stays engine-side). */
export interface PartSummary {
  id: string;
  name: string;
  materialKey: string;
  /** Bounding size in µm — the L×W×H shown in Properties. */
  sizeMicrons: { x: number; y: number; z: number };
}

export type ImportPhase = 'idle' | 'loading' | 'error';

export interface AppState {
  /** Selected part id (click a part or its Structure row). */
  selection: string | null;
  /** Active viewport mode; section/explode make their tools live. */
  viewMode: ViewMode;
  /** Live viewport frames-per-second (null until the engine reports). */
  fps: number | null;

  /** The imported model, as the panes see it. */
  parts: PartSummary[];
  modelName: string | null;
  importPhase: ImportPhase;
  importError: string | null;
  /**
   * True when the imported model's size looks implausible for a chip (wrong
   * declared units or up-axis are the usual culprits) — opens the review
   * dialog so the user can override both.
   */
  importNeedsReview: boolean;

  /** Section tool: which axis to cut across, and how deep (µm from +face). */
  sectionAxis: SectionAxis;
  sectionDepthMicrons: number;

  /** Explode tool: extra gap per stack tier, in µm. */
  explodeGapMicrons: number;

  /** The material color legend overlay. */
  legendVisible: boolean;

  /** Status-bar facts. */
  backendStatus: string | null;
  engineName: string | null;
}

let state: AppState = {
  selection: null,
  viewMode: 'orbit',
  fps: null,
  parts: [],
  modelName: null,
  importPhase: 'idle',
  importError: null,
  importNeedsReview: false,
  sectionAxis: 'y',
  sectionDepthMicrons: 0,
  explodeGapMicrons: 500,
  legendVisible: true,
  backendStatus: null,
  engineName: null,
};

const listeners = new Set<() => void>();

function emit() {
  for (const l of listeners) l();
}

export function getState(): AppState {
  return state;
}

export function setState(patch: Partial<AppState>) {
  state = { ...state, ...patch };
  emit();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Subscribe a component to a slice of app state. */
export function useAppState<T>(selector: (s: AppState) => T): T {
  return useSyncExternalStore(subscribe, () => selector(state));
}
