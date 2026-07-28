// engine/viewport.js
// Uses THREE.js for 3D visualizations.
// Anchor is the model center (or double click to select pivot).
// Keeps track of mouse position in the renderer's DOM element --- calculates deltas, decides how to rotate, etc.
// F hotkey = frame(), resets view to center and reset to default zooms and whatnot.
// Sets up the renderer, the camera view, functions that define how the view should react based off a click, etc.
// init() will be called in ViewportHost.tsx, where the Three.js engine will be turned into DOM element to be hooked up into frontend.

import * as THREE from 'three';

const MIN_ZOOM = 0.05;
const MAX_ZOOM = 60;

export function init(container, opts = {}) {
  /* Create renderer --> creates canvas --> insert canvas into container --> render scene + camera into canvas. */
  /* Container is section of div; you can document.getElementById(ID of the div container) to contain elements. */
  const { onFps = null } = opts; // If opts.onFps doesn't exist, use null.

  // Renderer instantiation --- talks to GPU and draws pixels onto canvas (the only DOM element to be added in).
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false }); // Smooth eges, opaque canvas.
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5)); // Cap pixel ratio for GPU.
  renderer.localClippingEnabled = true; // Slice geometry with clipping planes (sectioning for layers in future, etc.)
  renderer.domElement.style.display = 'block';
  renderer.domElement.style.touchAction = 'none';
  container.appendChild(renderer.domElement);

  // renderer.render(scene, camera)
  const scene = new THREE.Scene(); // The 3D representation.
  // Orthographic camera view.
  const camera = new THREE.OrthographicCamera(-30, 30, 20, -20, -2000, 4000);

  // Lighting setup for easier visuals.
  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(35, 80, 50);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xdfe8ff, 0.25);
  fill.position.set(-40, 20, -30);
  scene.add(fill);

  // Navigation state section for camera.
  const target = new THREE.Vector3(); // What the camera looks at.
  const anchor = new THREE.Vector3(); // What rotation revolves around.
  let lastBox = null; // Will later keep track of how big the model is.

  function size() {
    /* Size of viewport (the container). DIV width and length. */
    return {
      w: Math.max(1, container.clientWidth),
      h: Math.max(1, container.clientHeight),
    };
  }

  function modelRadius() {
    /* How big is model to determine camera positioning. */
    if (!lastBox) {
      return 20; // Default radius when no loaded model.
    }
    const size = lastBox.getSize(new THREE.Vector3());
    const width = size.x;
    const height = size.y;
    const depth = size.z;
    return Math.max(width, height, depth); // Return biggest dimension.
  }

  function updateOrthoFrustum() {
    /* Camera aspect of Three.JS convention for orthographic projection. Shows objects with no perspective distortion. */
    const { w, h } = size(); // Viewport size.
    const asp = w / h;
    const half = modelRadius() * 0.72;
    camera.left = -half * asp;
    camera.right = half * asp;
    camera.top = half;
    camera.bottom = -half;
    camera.updateProjectionMatrix();
  }

  function resize() {
    const { w, h } = size();
    renderer.setSize(w, h, true);
    updateOrthoFrustum();
  }

  // Camera basis columns (world space).
  const _right = new THREE.Vector3();
  const _upv = new THREE.Vector3();
  function basis() {
    camera.updateMatrixWorld();
    _right.setFromMatrixColumn(camera.matrixWorld, 0);
    _upv.setFromMatrixColumn(camera.matrixWorld, 1);
  }

  const _q = new THREE.Quaternion();
  /* Quaternion rotation around anchor (not around camera). */
  function applyRot(q) {
    camera.position.sub(anchor).applyQuaternion(q).add(anchor);
    target.sub(anchor).applyQuaternion(q).add(anchor);
    camera.up.applyQuaternion(q);
    camera.lookAt(target);
  }

  const _yAxis = new THREE.Vector3(0, 1, 0);
  function rotate(dxPx, dyPx) {
    const { h } = size();
    const speed = (2 * Math.PI) / h; // full drag height ≈ one revolution
    // yaw: about world Y through the anchor. No pole lock — you can tumble
    // clean over the top/bottom. When the view is upside down (up·Y < 0) the
    // yaw sign flips so horizontal drag keeps feeling natural.
    if (dxPx !== 0) {
      const s = camera.up.y >= 0 ? 1 : -1;
      applyRot(_q.setFromAxisAngle(_yAxis, -dxPx * speed * s));
    }
    // pitch: about the camera's right axis through the anchor, unclamped —
    // the rigid quaternion rotation crosses the poles without gimbal issues.
    if (dyPx !== 0) {
      basis();
      applyRot(_q.setFromAxisAngle(_right, -dyPx * speed));
    }
  }

  function pan(dxPx, dyPx) {
    basis();
    const { w, h } = size();
    const wppX = (camera.right - camera.left) / camera.zoom / w;
    const wppY = (camera.top - camera.bottom) / camera.zoom / h;
    const shift = _right
      .clone()
      .multiplyScalar(-dxPx * wppX)
      .add(_upv.clone().multiplyScalar(dyPx * wppY));
    camera.position.add(shift);
    target.add(shift);
    camera.lookAt(target);
  }

  function zoomAt(clientX, clientY, deltaY) {
    const rect = renderer.domElement.getBoundingClientRect();
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = -((clientY - rect.top) / rect.height) * 2 + 1;
    const zOld = camera.zoom;
    const zNew = THREE.MathUtils.clamp(
      zOld * Math.pow(0.998, deltaY),
      MIN_ZOOM,
      MAX_ZOOM,
    );
    if (zNew === zOld) return;
    // Keep the world point under the cursor stationary.
    basis();
    const halfW = (camera.right - camera.left) / 2;
    const halfH = (camera.top - camera.bottom) / 2;
    const k = 1 / zOld - 1 / zNew;
    const shift = _right
      .clone()
      .multiplyScalar(ndcX * halfW * k)
      .add(_upv.clone().multiplyScalar(ndcY * halfH * k));
    camera.zoom = zNew;
    camera.position.add(shift);
    target.add(shift);
    camera.updateProjectionMatrix();
    camera.lookAt(target);
  }

  /* Capture mouse actions on canvas, convert into camera rotation/panning/zooming. */
  let mode = null; // 'rotate' (left hold) or 'pan' (right hold)

  // Keep track of last mouse position for delta calculations.
  let lastX = 0;
  let lastY = 0;
  const el = renderer.domElement; // el = canvas element the renderer creates --> attaches mouse to canvas.

  function onPointerDown(e) {
    if (e.button === 0) { // Left hold.
      mode = 'rotate'
    }
    else if (e.button === 1 || e.button === 2) { // Middle/right hold.
      mode = 'pan'
    }
    else return;
    lastX = e.clientX;
    lastY = e.clientY;
    el.setPointerCapture(e.pointerId);
  }
  function onPointerMove(e) {
    if (!mode) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    if (mode === 'rotate') rotate(dx, dy);
    else pan(dx, dy);
  }
  function onPointerUp(e) {
    mode = null;
    if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
  }
  function onWheel(e) {
    e.preventDefault();
    zoomAt(e.clientX, e.clientY, e.deltaY);
  }
  const onContextMenu = (e) => e.preventDefault();

  // Three.js creates canvas. The canvas is a DOM element, (<div>, <canvas>, <button>, <span>) and all elements have addEventListener.
  // addEventListener browser events: pointerdown, pointermove, pointerup, pointercancel, wheel, contextmenu.
  el.addEventListener('pointerdown', onPointerDown); // Mouse button press.
  el.addEventListener('pointermove', onPointerMove); // Pointer moves while over element (drag).
  el.addEventListener('pointerup', onPointerUp); // Mouse button released.
  el.addEventListener('pointercancel', onPointerUp); // Window loses focus, pointercancel.
  el.addEventListener('wheel', onWheel, { passive: false }); // Scroll wheel.
  el.addEventListener('contextmenu', onContextMenu); // Prevents right click menu.

  const ISO_DIR = new THREE.Vector3(0.8, 0.85, 1.15).normalize();

  function frameBox(box) {
    if (!box || box.isEmpty()) {
      // Fallback bounding box for camera to frame if no uploaded model.
      box = new THREE.Box3(
        new THREE.Vector3(-10, 0, -10), // Min corner.
        new THREE.Vector3(10, 8, 10), // Max corner.
      );
    }
    lastBox = box.clone();
    const c = box.getCenter(new THREE.Vector3());
    anchor.copy(c);
    target.copy(c);
    const r = modelRadius();
    camera.up.set(0, 1, 0);
    camera.position.copy(c).add(ISO_DIR.clone().multiplyScalar(r * 1.8)); // Set camera view slightly larger than model radius.
    camera.zoom = 1;
    updateOrthoFrustum(); // Resize to fit viewport.
    camera.lookAt(target);
  }

  // Clears pan & pivot.
  function frame() {
    frameBox(lastBox);
  }

  // Raycaster --> shoots a ray from direction, mathematically checks inserctions with meshes.
  // Returns list of hit results sorted by distance, finds 3D point on surface it clicked.
  // Determines which object user clicked, where, etc.
  const _ray = new THREE.Raycaster();
  const _ndc = new THREE.Vector2();

  function setPivotFromClient(clientX, clientY) {
    const rect = renderer.domElement.getBoundingClientRect();
    _ndc.set( // _ndc = mouse pos.
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    _ray.setFromCamera(_ndc, camera);
    const hits = _ray.intersectObjects(scene.children, true); 
    if (hits.length) {
      anchor.copy(hits[0].point); // Change anchor (for future event listener).
    }
  }

  // Vectors to define standard CAD views.
  const VIEWS = {
    iso: { dir: [0.8, 0.85, 1.15], up: [0, 1, 0] },
    top: { dir: [0, 1, 0], up: [0, 0, -1] },
    bottom: { dir: [0, -1, 0], up: [0, 0, 1] },
    front: { dir: [0, 0, 1], up: [0, 1, 0] },
    back: { dir: [0, 0, -1], up: [0, 1, 0] },
    right: { dir: [1, 0, 0], up: [0, 1, 0] },
    left: { dir: [-1, 0, 0], up: [0, 1, 0] },
  };
  function setView(name) {
    const view = VIEWS[name] || VIEWS.iso;
    let center;
    if (lastBox) {
      center = lastBox.getCenter(new THREE.Vector3())
    } else {
      center = new THREE.Vector3();
    }
    anchor.copy(center);
    target.copy(center);
    const dist = Math.max(modelRadius(), 1) * 1.8;
    camera.up.set(view.up[0], view.up[1], view.up[2]);
    camera.position
      .copy(center)
      .add(new THREE.Vector3(view.dir[0], view.dir[1], view.dir[2]).normalize().multiplyScalar(dist));
    camera.zoom = 1;
    updateOrthoFrustum();
    camera.lookAt(target);
  }

  function setClearColor(cssColor) {
    renderer.setClearColor(new THREE.Color(cssColor.trim()), 1);
  }

  const observer = new ResizeObserver(resize);
  observer.observe(container);
  resize();

  // Render loop.
  let raf = 0;
  let frames = 0;
  let tPrev = performance.now();
  function loop(t) {
    raf = requestAnimationFrame(loop);
    renderer.render(scene, camera); // Scans all materials & objects (flags like needsUpdate recompiles that part).
    frames += 1;
    if (onFps && t - tPrev >= 500) {
      onFps((frames * 1000) / (t - tPrev));
      frames = 0;
      tPrev = t;
    }
  }
  raf = requestAnimationFrame(loop);

  function dispose() {
    cancelAnimationFrame(raf);
    observer.disconnect();
    el.removeEventListener('pointerdown', onPointerDown);
    el.removeEventListener('pointermove', onPointerMove);
    el.removeEventListener('pointerup', onPointerUp);
    el.removeEventListener('pointercancel', onPointerUp);
    el.removeEventListener('wheel', onWheel);
    el.removeEventListener('contextmenu', onContextMenu);
    scene.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach((m) => m.dispose());
      }
    });
    renderer.dispose();
    if (renderer.domElement.parentNode === container) {
      container.removeChild(renderer.domElement);
    }
  }

  // `controls` shim kept for API compatibility (target + update()).
  const controls = {
    target,
    update: () => camera.lookAt(target),
    dispose: () => {},
  };

  return {
    scene,
    camera,
    controls,
    renderer,
    resize,
    dispose,
    frameBox,
    frame,
    setPivotFromClient,
    setView,
    setClearColor,
    add: (obj) => scene.add(obj),
  };
}
