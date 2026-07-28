"""
cad
===
CAD ingestion for NerveGear: import a mesh (STL / OBJ / PLY / GLB), normalize it, and
convert it into the representation the CFD surrogate consumes -- a 128x128 binary
occupancy mask (1 = solid, 0 = fluid) on the unit domain the model was trained on.

This is the bridge between "user drops in a part" and "the physics model can see
it". The frontend renders the original mesh for manipulation; the backend turns
that same mesh into the obstacle mask that /predict expects.

Heavy dependencies (trimesh) are imported inside the submodules, not here, so
importing `cad` is cheap.
"""

from .ingest import MeshInfo, ingest, load_mesh, normalize
from .voxelize import mesh_to_mask

__all__ = ["MeshInfo", "ingest", "load_mesh", "normalize", "mesh_to_mask"]
