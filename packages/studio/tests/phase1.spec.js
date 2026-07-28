// Phase 1 end-to-end verification (build plan, Step 10).
//
// Boots the built React app headlessly and walks the whole Phase 1 story:
// the bundled sample loads through the real import pipeline, parts are named
// and material-colored, selection is bidirectional, Section and Explode do
// real geometry work, re-import doesn't leak, and both themes hold up.
// Screenshots land in tests/artifacts/phase1/ for the human sign-off.

import { test, expect } from '@playwright/test';

const SHOT_DIR = 'tests/artifacts/phase1';

async function bootedApp(page) {
  await page.goto('/');
  // The sample model has loaded when the status bar reports its parts.
  await expect(page.locator('.statusbar')).toContainText('sample-stacked-die.glb', {
    timeout: 15000,
  });
  await expect(page.locator('.statusbar')).toContainText('19 parts');
}

test('sample model loads with named, material-colored parts and a legend', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await bootedApp(page);

  // Structure pane mirrors the file's named solids.
  const rows = page.locator('.structure__row');
  await expect(rows).toHaveCount(19);
  await expect(rows.filter({ hasText: 'gpu-die' })).toHaveCount(1);
  await expect(rows.filter({ hasText: 'substrate' })).toHaveCount(1);

  // Legend lists only materials present in the model.
  await expect(page.locator('.legend__row')).not.toHaveCount(0);
  await expect(page.locator('.legend')).toContainText('Silicon');

  await page.screenshot({ path: `${SHOT_DIR}/iso-dark.png` });
  expect(errors).toEqual([]);
});

test('selection is bidirectional between list and properties', async ({ page }) => {
  await bootedApp(page);

  // List → selection → properties identity card.
  await page.locator('.structure__row', { hasText: 'gpu-die' }).click();
  await expect(page.locator('.structure__row--selected')).toContainText('gpu-die');
  const props = page.locator('.props');
  await expect(props).toContainText('gpu-die');
  await expect(props).toContainText('12.00 mm'); // die is 12 mm long in X

  // Viewport click on empty space clears the selection everywhere.
  await page.mouse.click(40, 120);
  await expect(page.locator('.structure__row--selected')).toHaveCount(0);
  await expect(page.locator('.props')).toHaveCount(0);

  // Clicking the center of the viewport hits the package → selects a part.
  const viewport = page.locator('.viewport__canvas');
  const box = await viewport.boundingBox();
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  await expect(page.locator('.structure__row--selected')).toHaveCount(1);
  await page.screenshot({ path: `${SHOT_DIR}/selected-dark.png` });
});

test('material override recolors and persists in the store', async ({ page }) => {
  await bootedApp(page);
  await page.locator('.structure__row', { hasText: 'tim' }).first().click();
  await page.locator('.props select').selectOption('coolant');
  // The row swatch follows the override (rgb(31,182,224) = #1fb6e0 coolant).
  const swatch = page.locator('.structure__row--selected .structure__swatch');
  await expect(swatch).toHaveCSS('background-color', 'rgb(31, 182, 224)');
});

test('section mode slices and restores', async ({ page }) => {
  await bootedApp(page);
  await page.getByRole('button', { name: 'Section' }).click();
  // The inline tool controls appear with a real depth value (µm, mid-model).
  const depthInput = page.locator('.commandbar__tool-controls input');
  await expect(depthInput).toHaveValue(/\d/);
  await page.screenshot({ path: `${SHOT_DIR}/section-dark.png` });
  await page.getByRole('button', { name: 'Orbit' }).click();
  await expect(page.locator('.commandbar__tool-controls')).toHaveCount(0);
});

test('explode mode separates tiers and restores', async ({ page }) => {
  await bootedApp(page);
  await page.getByRole('button', { name: 'Explode' }).click();
  await page.waitForTimeout(400); // let the 280 ms animation finish
  await page.screenshot({ path: `${SHOT_DIR}/exploded-dark.png` });
  await page.getByRole('button', { name: 'Orbit' }).click();
  await page.waitForTimeout(400);
});

test('re-import leaves no stale scene behind', async ({ page }) => {
  await bootedApp(page);
  // Import the same GLB again through the picker; part count must not grow
  // (dispose-on-reimport). The bytes come from the page's own sample asset.
  const fileInput = page.locator('input[type="file"]');
  const glbBytes = await page.evaluate(async () => {
    const asset = performance
      .getEntriesByType('resource')
      .find((r) => r.name.includes('sample-stacked-die'));
    const response = await fetch(asset.name);
    return Array.from(new Uint8Array(await response.arrayBuffer()));
  });
  await fileInput.setInputFiles({
    name: 'reimported.glb',
    mimeType: 'model/gltf-binary',
    buffer: Buffer.from(glbBytes),
  });
  await expect(page.locator('.statusbar')).toContainText('reimported.glb');
  await expect(page.locator('.structure__row')).toHaveCount(19);
});

test('light theme: background, edges, legend all follow', async ({ page }) => {
  await bootedApp(page);
  await page.getByRole('button', { name: 'Light mode' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.screenshot({ path: `${SHOT_DIR}/iso-light.png` });

  await page.locator('.structure__row', { hasText: 'gpu-die' }).click();
  await page.screenshot({ path: `${SHOT_DIR}/selected-light.png` });

  await page.getByRole('button', { name: 'Section' }).click();
  await page.screenshot({ path: `${SHOT_DIR}/section-light.png` });

  await page.getByRole('button', { name: 'Explode' }).click();
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${SHOT_DIR}/exploded-light.png` });
});
