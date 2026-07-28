// engine/model/explode.ts — the Explode tool: pull the stack apart vertically.

import * as THREE from 'three';
import type { Part } from './part';
import type { ChipScene } from './chipScene';

// Two parts whose bottom faces are within this distance belong to one tier.
// 5 µm absorbs tessellation noise but never merges real layers (the thinnest
// real features — bump/TIM layers — are tens of µm).
const TIER_MERGE_TOLERANCE_MICRONS = 5;

const ANIMATION_DURATION_MS = 280;

export interface ExplodeController {
  /** Animate toward the given state. Amount = gap added per tier, in µm. */
  setExploded(exploded: boolean, gapPerTierMicrons: number): void;
  /** Jump to the current target instantly (used on re-import). */
  dispose(): void;
}

export function createExplodeController(
  parts: Part[],
  chipScene: ChipScene,
): ExplodeController {
  const tierByPartId = assignStackTiers(parts);

  let animationFrame = 0;
  let currentFactor = 0; // 0 = assembled, 1 = fully exploded
  let targetFactor = 0;
  let gapMicrons = 0;

  function applyFactor(factor: number): void {
    currentFactor = factor;
    for (const [partId, tier] of tierByPartId) {
      const mesh = chipScene.meshByPartId.get(partId);
      // Geometry is baked in place, so position is a pure explode offset (µm).
      if (mesh) mesh.position.y = tier * gapMicrons * factor;
    }
  }

  function animateToTarget(): void {
    cancelAnimationFrame(animationFrame);
    const startFactor = currentFactor;
    const startTime = performance.now();

    function step(now: number): void {
      const progress = Math.min((now - startTime) / ANIMATION_DURATION_MS, 1);
      const eased = easeInOut(progress);
      applyFactor(startFactor + (targetFactor - startFactor) * eased);
      if (progress < 1) animationFrame = requestAnimationFrame(step);
    }
    animationFrame = requestAnimationFrame(step);
  }

  return {
    setExploded(exploded, gapPerTierMicrons) {
      gapMicrons = gapPerTierMicrons;
      targetFactor = exploded ? 1 : 0;
      // Re-apply immediately if only the gap changed while exploded.
      if (currentFactor === targetFactor) applyFactor(currentFactor);
      else animateToTarget();
    },
    dispose() {
      cancelAnimationFrame(animationFrame);
      applyFactor(0);
    },
  };
}

/**
 * Stack order from the geometry itself: sort parts by the bottom of their
 * bounding box, then group near-equal bottoms into tiers 0, 1, 2, …
 * Tier 0 (the substrate) never moves; each higher tier gains one more gap.
 */
function assignStackTiers(parts: Part[]): Map<string, number> {
  const bottomsSorted = [...parts].sort(
    (a, b) => a.boundingBox.min.y - b.boundingBox.min.y,
  );

  const tierByPartId = new Map<string, number>();
  let tier = -1;
  let previousBottom = Number.NEGATIVE_INFINITY;
  for (const part of bottomsSorted) {
    const bottom = part.boundingBox.min.y;
    if (bottom - previousBottom > TIER_MERGE_TOLERANCE_MICRONS) {
      tier += 1;
      previousBottom = bottom;
    }
    tierByPartId.set(part.id, tier);
  }
  return tierByPartId;
}

function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}
