/*
React component — the bridge between the Three.js engine and the React UI.
It owns no visual logic of its own: the engine modules (viewport, chipScene,
selection, section, explode) do the 3D work, the store carries the UI state,
and this file just keeps the two sides agreeing:

  model changes (modelSession) ──► rebuild scene, frame it, reset tools
  store changes (selection, tools, theme) ──► drive the engine
  canvas clicks ──► pick a part ──► write store.selection
*/

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { useTheme } from '../theme/ThemeProvider';
import { palette } from '../theme/tokens';
import { setState, getState, useAppState } from '../state/store';
import { onModelChange } from '../state/modelSession';

import { init } from '../../engine/viewport.js';
import { buildChipScene, type ChipScene } from '../../engine/model/chipScene';
import {
  pickPartAt,
  createSelectionHighlighter,
  type SelectionHighlighter,
} from '../../engine/model/selection';
import { createSectionController, type SectionController } from '../../engine/model/section';
import { createExplodeController, type ExplodeController } from '../../engine/model/explode';
import { RENDER_UNITS_PER_MICRON } from '../../engine/model/part';
import { ViewGizmo, type ViewName } from './ViewGizmo';
import { LegendOverlay } from './LegendOverlay';

type Viewport = ReturnType<typeof init>;

// A pointer that moves less than this between down and up is a click (a selection), not an orbit drag.
const CLICK_SLOP_PIXELS = 5;

export function ViewportHost() {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<Viewport | null>(null);
  const chipSceneRef = useRef<ChipScene | null>(null);
  const highlighterRef = useRef<SelectionHighlighter | null>(null);
  const sectionRef = useRef<SectionController | null>(null);
  const explodeRef = useRef<ExplodeController | null>(null);
  const modelBoundsMicronsRef = useRef(new THREE.Box3());
  const { theme } = useTheme();

  // engine lifetime: one viewport + tool controllers per mount 
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const vp = init(el, { onFps: (fps: number) => setState({ fps }) });
    viewportRef.current = vp;

    const section = createSectionController();
    sectionRef.current = section;
    vp.add(section.group);
    section.attach(vp.renderer.domElement, vp.camera, (depthMicrons: number) =>
      setState({ sectionDepthMicrons: Math.round(depthMicrons) }),
    );

    // Rebuild the 3D scene whenever a model is imported (fires immediately with the current model, so a mount after import still shows it).
    const stopModelListener = onModelChange((model) => {
      chipSceneRef.current?.dispose();
      highlighterRef.current?.dispose();
      explodeRef.current?.dispose();
      chipSceneRef.current = null;
      highlighterRef.current = null;
      explodeRef.current = null;
      if (!model) return;

      const chipScene = buildChipScene(model.parts);
      chipScene.setEdgeColor(edgeColorFor());
      chipSceneRef.current = chipScene;
      vp.add(chipScene.root);

      const highlighter = createSelectionHighlighter(chipScene, palette[themeNow()].selection);
      highlighterRef.current = highlighter;
      explodeRef.current = createExplodeController(model.parts, chipScene);

      modelBoundsMicronsRef.current = model.boundingBox.clone();
      vp.frameBox(renderBoundsOf(model.boundingBox));
      applyToolState();
    });

    // Click (not drag) → select the part under the cursor.
    const downAt = { x: 0, y: 0 };
    const onPointerDown = (e: PointerEvent) => {
      downAt.x = e.clientX;
      downAt.y = e.clientY;
    };
    const onPointerUp = (e: PointerEvent) => {
      if (e.button !== 0 || sectionRef.current?.isDragging()) return;
      const moved = Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y);
      if (moved > CLICK_SLOP_PIXELS) return;
      const chipRoot = chipSceneRef.current?.root;
      if (!chipRoot) return;
      const partId = pickPartAt(e.clientX, e.clientY, vp.renderer.domElement, vp.camera, chipRoot);
      setState({ selection: partId });
    };
    vp.renderer.domElement.addEventListener('pointerdown', onPointerDown);
    vp.renderer.domElement.addEventListener('pointerup', onPointerUp);

    const onDbl = (e: MouseEvent) => vp.setPivotFromClient(e.clientX, e.clientY);
    el.addEventListener('dblclick', onDbl);
    const onKey = (e: KeyboardEvent) => {
      if ((e.key === 'f' || e.key === 'F') && !e.repeat) vp.frame();
    };
    window.addEventListener('keydown', onKey);

    return () => {
      stopModelListener();
      el.removeEventListener('dblclick', onDbl);
      window.removeEventListener('keydown', onKey);
      vp.renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      vp.renderer.domElement.removeEventListener('pointerup', onPointerUp);
      chipSceneRef.current?.dispose();
      highlighterRef.current?.dispose();
      explodeRef.current?.dispose();
      sectionRef.current?.dispose();
      viewportRef.current = null;
      setState({ fps: null });
      vp.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── store → engine: each slice drives one engine feature ──
  const selection = useAppState((s) => s.selection);
  const viewMode = useAppState((s) => s.viewMode);
  const sectionAxis = useAppState((s) => s.sectionAxis);
  const sectionDepthMicrons = useAppState((s) => s.sectionDepthMicrons);
  const explodeGapMicrons = useAppState((s) => s.explodeGapMicrons);
  const parts = useAppState((s) => s.parts);
  const legendVisible = useAppState((s) => s.legendVisible);

  useEffect(() => {
    highlighterRef.current?.highlight(selection);
  }, [selection, parts]);

  useEffect(() => {
    applyToolState();
    // Entering Section with no depth set yet: start the cut mid-model so the
    // first click visibly slices instead of doing nothing.
    if (viewMode === 'section' && sectionDepthMicrons === 0) {
      const bounds = modelBoundsMicronsRef.current;
      if (!bounds.isEmpty()) {
        const span = bounds.max[sectionAxis] - bounds.min[sectionAxis];
        setState({ sectionDepthMicrons: Math.round(span / 2) });
      }
    }
  }, [viewMode, sectionAxis, sectionDepthMicrons, explodeGapMicrons]);

  // Manual material overrides: recolor the live meshes to match the store.
  useEffect(() => {
    const chipScene = chipSceneRef.current;
    if (!chipScene) return;
    for (const part of parts) chipScene.setPartMaterial(part.id, part.materialKey);
  }, [parts]);

  // Theme: background, feature-edge color, and accent outline all follow it.
  useEffect(() => {
    viewportRef.current?.setClearColor(palette[theme].bgApp);
    chipSceneRef.current?.setEdgeColor(edgeColorFor());
    highlighterRef.current?.setAccentColor(palette[theme].selection);
  }, [theme, parts]);

  /** Push the current view-mode/tool values into the section & explode engines. */
  function applyToolState() {
    const s = getState();
    const bounds = modelBoundsMicronsRef.current;

    const placement =
      s.viewMode === 'section'
        ? { axis: s.sectionAxis, depthMicrons: s.sectionDepthMicrons }
        : null;
    const planes = sectionRef.current?.apply(placement, bounds) ?? [];
    chipSceneRef.current?.setClippingPlanes(planes);
    highlighterRef.current?.setClippingPlanes(planes);

    explodeRef.current?.setExploded(s.viewMode === 'explode', s.explodeGapMicrons);
  }

  function edgeColorFor(): string {
    return themeNow() === 'dark' ? '#8a97a8' : '#1a2027';
  }

  function themeNow(): 'dark' | 'light' {
    return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
  }

  const applyView = (name: ViewName) => viewportRef.current?.setView(name);
  const reframe = () => viewportRef.current?.frame();

  return (
    <div className="viewport">
      <div ref={containerRef} className="viewport__canvas" />
      <div className="viewport__hud">
        <div className="viewport__hud-corner viewport__hud-corner--tr">
          <ViewGizmo onView={applyView} onReframe={reframe} />
        </div>
        {legendVisible && (
          <div className="viewport__hud-corner viewport__hud-corner--br">
            <LegendOverlay />
          </div>
        )}
        <div className="viewport__hud-corner viewport__hud-corner--bl">
          <span className="viewport__hint">
            click select · L-drag rotate · R/M-drag pan · scroll zoom · dbl-click pivot · F fit
          </span>
        </div>
      </div>
    </div>
  );
}

/** Model bounds are in µm; the camera lives in render units (mm). */
function renderBoundsOf(boundsMicrons: THREE.Box3): THREE.Box3 {
  return new THREE.Box3(
    boundsMicrons.min.clone().multiplyScalar(RENDER_UNITS_PER_MICRON),
    boundsMicrons.max.clone().multiplyScalar(RENDER_UNITS_PER_MICRON),
  );
}
