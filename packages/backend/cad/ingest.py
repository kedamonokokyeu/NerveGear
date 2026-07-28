"""
cad/ingest.py

Loads a CAD mesh and put it into a normalized frame so everything
downstream (rendering, voxelization, physics) can follow the same conventions.

Normalize = center the mesh on origin and uniformly scale it so the largest dimension is 1.0

Supported formats are whatever trimesh can read: STL, OBJ, PLY, GLB/GLTF, OFF.
STEP/IGES (BREP) are not handled here.

Trimesh = Python library for loading, analyzing, and manipulating 3D triangle
meshes. Reads CAD-style files and gives arrays of vertices, faces, etc.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class MeshInfo:
    """Human- and JSON-friendly summary of an ingested mesh."""

    source: str
    n_vertices: int
    n_faces: int
    watertight: bool          # closed surface matters for volume + reliable voxel fill
    bbox_min: tuple
    bbox_max: tuple
    extents: tuple            # bounding-box size along x, y, z (post-normalization)
    volume: float             # only meaningful if watertight
    area: float
    scale_applied: float      # the uniform factor used to normalize (orig -> unit)

    def to_dict(self) -> dict:
        return asdict(self)


def load_mesh(path: str):
    """
    Load a mesh file into a single trimesh.Trimesh. Scenes with multiple
    meshes (common in OBJ/GLB) are concatenated into one body.
    Returns a mesh.
    """
    import trimesh  # lazy: keeps the dependency out of the import path until needed

    obj = trimesh.load(path, force="mesh") # always return a mesh
    if isinstance(obj, trimesh.Scene): # if a Scene (collection of meshes) then merge the meshes into one
        obj = trimesh.util.concatenate(tuple(obj.geometry.values()))
    if not isinstance(obj, trimesh.Trimesh) or obj.faces.shape[0] == 0: # checks if mesh has triangles
        raise ValueError(f"No triangulated geometry found in {path!r}")
    return obj


def normalize(mesh):
    """
    Return (new_mesh, scale) with the mesh centered at the origin. Scaled so
    its largest bounding-box dimension is exactly 1.0. The original is not mutated.
    """
    m = mesh.copy()
    m.apply_translation(-m.bounding_box.centroid)  # center on origin
    extent = float(m.extents.max())
    scale = 1.0 / extent if extent > 0 else 1.0
    m.apply_scale(scale)
    return m, scale


def ingest(path: str):
    """
    Full pipeline: load -> normalize -> summarize.
    Returns (normalized_mesh, MeshInfo).
    """
    raw = load_mesh(path)
    mesh, scale = normalize(raw)
    info = MeshInfo(
        source=str(path),
        n_vertices=int(mesh.vertices.shape[0]),
        n_faces=int(mesh.faces.shape[0]),
        watertight=bool(mesh.is_watertight),
        bbox_min=tuple(np.round(mesh.bounds[0], 6).tolist()),
        bbox_max=tuple(np.round(mesh.bounds[1], 6).tolist()),
        extents=tuple(np.round(mesh.extents, 6).tolist()),
        volume=float(mesh.volume) if mesh.is_watertight else float("nan"),
        area=float(mesh.area),
        scale_applied=float(scale),
    )
    return mesh, info
