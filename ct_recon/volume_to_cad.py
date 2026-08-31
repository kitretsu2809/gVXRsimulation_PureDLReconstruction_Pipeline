import argparse
import os
import warnings

import numpy as np
import tifffile
import trimesh
from skimage import measure
from skimage.filters import threshold_otsu


def load_tiff_volume(path: str) -> np.ndarray:
    """
    Load a 3D TIFF volume using tifffile.

    Args:
        path (str): Path to the TIFF file.

    Returns:
        np.ndarray: The 3D numpy array containing the volume data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    
    print(f"Loading volume from {path}...")
    volume = tifffile.imread(path)
    print(f"Loaded volume with shape {volume.shape} and dtype {volume.dtype}")
    return volume


def export_volume_to_cad(
    volume: np.ndarray,
    voxel_spacing: tuple = (0.5, 0.5, 0.5),  # mm
    isovalue: float = None,  # None = auto Otsu
    output_stl_path: str = 'reconstruction.stl',
    output_obj_path: str = None,  # Optional OBJ export
    smooth_iterations: int = 10,
    decimate_fraction: float = 0.5,  # Reduce mesh by 50%
) -> trimesh.Trimesh:
    """
    Convert a 3D numpy voxel volume into CAD-ready mesh files (STL, OBJ).

    Args:
        volume (np.ndarray): 3D volume data.
        voxel_spacing (tuple, optional): Voxel spacing in (x, y, z) dimensions. Defaults to (0.5, 0.5, 0.5).
        isovalue (float, optional): Isovalue for surface extraction. If None, Otsu's thresholding is used. Defaults to None.
        output_stl_path (str, optional): Path to save the STL file. Defaults to 'reconstruction.stl'.
        output_obj_path (str, optional): Path to save the OBJ file. Defaults to None.
        smooth_iterations (int, optional): Number of Laplacian smoothing iterations. Defaults to 10.
        decimate_fraction (float, optional): Target fraction of faces to keep (0.0 to 1.0). Defaults to 0.5.

    Returns:
        trimesh.Trimesh: The processed trimesh mesh object.
    """
    # Auto-detect isovalue using Otsu thresholding if not provided
    if isovalue is None:
        vol_std = float(volume.std())
        vol_range = float(volume.max() - volume.min())
        if vol_range < 0.05 or vol_std < 0.01:
            print(f"⚠️  WARNING: Volume has near-zero dynamic range (std={vol_std:.4f}, range={vol_range:.4f}).")
            print("   This means the model failed to reconstruct object structure.")
            print("   The model needs to be retrained. Skipping CAD export.")
            raise ValueError(
                f"Volume has near-zero dynamic range (std={vol_std:.4f}). "
                "The model output is a uniform gray blob — no object surface can be extracted. "
                "Please retrain the model with the weighted foreground loss fix."
            )
        print("Auto-detecting isovalue using Otsu's method...")
        isovalue = threshold_otsu(volume)
        print(f"Computed Otsu isovalue: {isovalue:.4f}")
    else:
        print(f"Using provided isovalue: {isovalue}")

    # Marching cubes for isosurface extraction
    print("Extracting isosurface using Marching Cubes...")
    verts, faces, normals, values = measure.marching_cubes(
        volume, level=isovalue, spacing=voxel_spacing, method='lewiner'
    )

    # Create trimesh object
    print(f"Generated mesh with {len(verts)} vertices and {len(faces)} faces.")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)

    # Mesh repair
    if not mesh.is_watertight:
        print("Mesh is not watertight. Attempting basic repairs...")
        mesh.fill_holes()

    # Laplacian smoothing to remove voxel staircase artifacts
    if smooth_iterations > 0:
        print(f"Applying Laplacian smoothing ({smooth_iterations} iterations)...")
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations)

    # Mesh decimation
    if decimate_fraction is not None and 0.0 < decimate_fraction < 1.0:
        target_faces = int(len(mesh.faces) * decimate_fraction)
        print(f"Decimating mesh to {target_faces} faces ({decimate_fraction*100:.1f}%)...")
        try:
            if hasattr(mesh, "simplify_quadric_decimation"):
                mesh = mesh.simplify_quadric_decimation(target_faces)
            elif hasattr(mesh, "simplify_quadratic_decimation"):
                mesh = mesh.simplify_quadratic_decimation(target_faces)
            print(f"Decimated mesh now has {len(mesh.faces)} faces.")
        except Exception as e:
            warnings.warn(f"Mesh decimation skipped ({e}). Keeping full-resolution mesh.")

    # Export to STL
    if output_stl_path:
        print(f"Exporting to binary STL: {output_stl_path}")
        mesh.export(output_stl_path, file_type='stl')

    # Export to OBJ
    if output_obj_path:
        print(f"Exporting to OBJ: {output_obj_path}")
        mesh.export(output_obj_path, file_type='obj')

    return mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert 3D CT volume to CAD mesh (STL/OBJ)')
    parser.add_argument('input', help='Path to 3D TIFF volume')
    parser.add_argument('--output-stl', default='reconstruction.stl', help='Path for output STL file (default: reconstruction.stl)')
    parser.add_argument('--output-obj', default=None, help='Path for output OBJ file (optional)')
    parser.add_argument('--isovalue', type=float, default=None, help='Isovalue for marching cubes. Auto-detected using Otsu if not specified.')
    parser.add_argument('--smooth', type=int, default=10, help='Number of Laplacian smoothing iterations (default: 10)')
    parser.add_argument('--decimate', type=float, default=0.5, help='Fraction of faces to keep during decimation, between 0 and 1 (default: 0.5)')
    parser.add_argument('--voxel-size', type=float, default=0.5, help='Voxel size/spacing in mm (default: 0.5)')

    args = parser.parse_args()

    # Create output directories if they don't exist
    for output_path in [args.output_stl, args.output_obj]:
        if output_path:
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)

    # Load volume
    volume = load_tiff_volume(args.input)

    # Spacing from command line argument
    spacing = (args.voxel_size, args.voxel_size, args.voxel_size)

    # Convert to CAD
    export_volume_to_cad(
        volume=volume,
        voxel_spacing=spacing,
        isovalue=args.isovalue,
        output_stl_path=args.output_stl,
        output_obj_path=args.output_obj,
        smooth_iterations=args.smooth,
        decimate_fraction=args.decimate
    )
    
    print("Conversion complete.")
