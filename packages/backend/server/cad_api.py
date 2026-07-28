"""
server/cad_api.py

The CAD-conversion HTTP endpoint the Studio's Import button talks to.

One route:  POST /cad/convert/step-to-glb
  in:  a .step/.stp upload (multipart form field "file")
  out: GLB bytes (the browser-native 3D format), plus a response header
       X-NerveGear-Microns-Per-Unit telling the viewer what one model unit
       means in µm (read from the STEP file's own header).

Kept separate from design_api.py on purpose: that file is the physics
contract (solve.v1); this file is file-format plumbing. They evolve
independently.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from cad.step_to_glb import convert_step_to_glb, read_step_length_unit_microns

logger = logging.getLogger("nervegear.cad")

router = APIRouter()

ACCEPTED_EXTENSIONS = (".step", ".stp")

# Refuse uploads past this size before tessellating. Matches the app-wide
# 40 MB body guard in server/app.py — a 40 MB STEP is already far beyond any
# die/package model this tool targets.
MAX_UPLOAD_BYTES = 40 * 1024 * 1024


@router.post("/cad/convert/step-to-glb")
async def step_to_glb_endpoint(file: UploadFile = File(...)) -> Response:
    filename = file.filename or "upload"
    if not filename.lower().endswith(ACCEPTED_EXTENSIONS):
        raise HTTPException(
            status_code=415,
            detail=f"Expected a {' / '.join(ACCEPTED_EXTENSIONS)} file, got '{filename}'.",
        )

    step_bytes = await file.read()
    if len(step_bytes) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(step_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB conversion limit.",
        )

    # Only for the log line: OpenCASCADE reads this same declared unit and
    # normalises the output GLB to glTF-spec METRES (verified empirically:
    # a 30 mm STEP box comes out 0.030 GLB units). So the unit we report to
    # the viewer is the constant below, not the STEP's own unit.
    declared_unit_microns = read_step_length_unit_microns(step_bytes)
    GLB_MICRONS_PER_UNIT = 1_000_000.0  # metres, per the glTF spec

    try:
        glb_bytes = convert_step_to_glb(step_bytes)
    except HTTPException:
        raise
    except Exception as error:  # OpenCASCADE failures are C++-side and varied
        logger.exception("STEP->GLB conversion failed for %s", filename)
        raise HTTPException(
            status_code=422,
            detail=f"Could not convert '{filename}': {error}",
        ) from error

    logger.info(
        "Converted %s: %d bytes STEP (declared 1 unit = %g um) -> %d bytes GLB",
        filename, len(step_bytes), declared_unit_microns, len(glb_bytes),
    )
    return Response(
        content=glb_bytes,
        media_type="model/gltf-binary",
        headers={"X-NerveGear-Microns-Per-Unit": str(GLB_MICRONS_PER_UNIT)},
    )
