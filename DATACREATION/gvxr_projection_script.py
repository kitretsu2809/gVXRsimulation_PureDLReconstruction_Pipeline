import os
import sys
import numpy as np
import argparse
from skimage.io import imsave
import trimesh

def preprocess_mesh(stl_filepath, temp_stl_path, safe_fov, scan_method):
    print(f"Preprocessing mesh with trimesh to guarantee alignment...")
    mesh = trimesh.load(stl_filepath, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded geometry from {stl_filepath} does not contain mesh vertices.")
    
    # 1. Center the mesh perfectly at (0,0,0)
    mesh.vertices -= mesh.bounding_box.centroid
    
    # 2. Align longest axis to Z (so it stands upright)
    extents = mesh.extents
    longest = np.argmax(extents)
    if longest == 0:
        matrix = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
        mesh.apply_transform(matrix)
    elif longest == 1:
        matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        mesh.apply_transform(matrix)
        
    # 3. Align shortest axis to Y (so the large flat face initially faces the X-ray source)
    extents = mesh.extents
    if extents[0] < extents[1]:
        matrix = trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1])
        mesh.apply_transform(matrix)
        
    # 4. Scale to fit FOV (Accounting for BOTH xy_diagonal and height Z)
    extents = mesh.extents
    xy_diagonal = np.sqrt(extents[0]**2 + extents[1]**2)
    max_dim = max(xy_diagonal, extents[2])
    
    desired_max_diameter = (safe_fov * 1.9) if scan_method == 'offset' else safe_fov
    
    if max_dim > desired_max_diameter:
        scale_factor = desired_max_diameter / max_dim
        mesh.apply_scale(scale_factor)
        print(f"  -> Scaled mesh down by {scale_factor:.4f} so it fits within {desired_max_diameter:.1f}mm FOV.")
    else:
        print(f"  -> Mesh size perfectly fits within the {scan_method} scanner FOV.")
        
    # Export the perfectly prepared mesh
    mesh.export(temp_stl_path)
    return max_dim

def run_gvxr_pipeline(stl_filepath, output_dir, material="Ti", i0=50000.0, gaussian_std=10.0, scan_method='auto'):
    from gvxrPython3 import gvxr

    # 1. Physics Engine Setup
    # X-ray source configuration matching Nikon XT H 225 ST 2x
    sod_mm = 300.0
    sdd_mm = 1100.0
    
    # 2. Detector Setup
    # The detector pixel size is 0.2mm. We want a ~250mm FOV, so we need a large detector
    # To cover a 250mm object diagonally, max_diagonal = ~350mm -> 1750x1750.
    det_size_pixels = 1800
    det_pixel_mm = 0.200

    print("Initializing gVirtualXray (gVXR) engine...")
    gvxr.createWindow()
    gvxr.setWindowSize(det_size_pixels, det_size_pixels)

    gvxr.setSourcePosition(0.0, -sod_mm, 0.0, "mm")
    gvxr.usePointSource()
    gvxr.setMonoChromatic(100.0, "keV", 1000)

    gvxr.setDetectorPosition(0.0, sdd_mm - sod_mm, 0.0, "mm")
    gvxr.setDetectorUpVector(0, 0, -1)
    gvxr.setDetectorNumberOfPixels(det_size_pixels, det_size_pixels)
    gvxr.setDetectorPixelSize(det_pixel_mm, det_pixel_mm, "mm")

    # Calculate true FOV at the origin considering magnification
    magnification = sdd_mm / sod_mm
    detector_width_mm = det_size_pixels * det_pixel_mm
    fov_at_origin = detector_width_mm / magnification
    safe_fov = fov_at_origin * 0.8 

    # Pre-process the mesh using our robust trimesh function
    if not output_dir.endswith(f"_{scan_method}"):
        output_dir = f"{output_dir}_{scan_method}"
    os.makedirs(output_dir, exist_ok=True)
    temp_stl_path = os.path.join(output_dir, "temp_aligned.stl")
    
    preprocess_mesh(stl_filepath, temp_stl_path, safe_fov, scan_method)

    print(f"Loading prepared mesh from {temp_stl_path}...")
    gvxr.loadMeshFile("object", temp_stl_path, "mm")
    print(f"Applying material properties ({material})...")
    gvxr.setElement("object", material)

    # Apply Offset if necessary
    if scan_method == 'offset':
        offset_pixels = (det_size_pixels / 2.0) - 20
        offset_mm = offset_pixels * det_pixel_mm
        gvxr.setDetectorPosition(offset_mm, sdd_mm - sod_mm, 0.0, "mm")
        print("gVXR Geometry set to OFF-CENTER Cone Beam.")
    else:
        print("gVXR Geometry set to CENTERED Cone Beam.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    projections_dir = os.path.join(output_dir, "projections")
    os.makedirs(projections_dir, exist_ok=True)

    print("Simulating X-ray attenuation projection sweeps using gVXR GPU Raytracing...")
    
    angles = np.arange(0, 360, 1)
    
    sino_max_list = []
    
    for i, angle in enumerate(angles):
        filename = os.path.join(projections_dir, f"projection_angle_{i:03d}.tiff")
        if os.path.exists(filename):
            print(f"Skipping projection {i:03d} at angle {angle}, already exists.")
            continue
            
        # Rotate object around Z axis
        gvxr.rotateNode("object", float(angle), 0, 0, 1)
        
        # Compute raw X-ray energy deposited on detector
        image = np.array(gvxr.computeXRayImage())
        
        # The background (unattenuated X-rays) receives maximum energy.
        # This acts as our Flat Field (I0).
        E0 = np.max(image)
        if E0 <= 0: E0 = 1.0
        
        # Calculate ideal transmission ratio in [0.0, 1.0]
        transmission_ratio = image / E0
        
        # Convert to expected photon count at the detector
        expected_photons = transmission_ratio * i0
        
        # Apply physical Poisson noise (photon starvation)
        noisy_photons = np.random.poisson(np.clip(expected_photons, 0, None))
        
        # Apply electronic sensor readout noise (Gaussian)
        noisy_photons = noisy_photons + np.random.normal(0, gaussian_std, noisy_photons.shape)
        
        # Prevent log(0) or negative photons
        noisy_photons = np.clip(noisy_photons, 1, None)
        
        # Calculate attenuation (Beer-Lambert Law): A = -ln(I / I0)
        projections = -np.log(noisy_photons / i0)
        
        # Calculate max for this specific projection
        sino_max = np.max(projections)
        if sino_max <= 0: sino_max = 1e-6
        sino_max_list.append(sino_max)
        
        # Scale to 16-bit
        normalized = projections / sino_max
        final_image = (np.clip(normalized, 0, 1) * 65535).astype(np.uint16)
        imsave(filename, final_image, check_contrast=False)
        
        # Restore rotation (undo cumulative rotation)
        gvxr.rotateNode("object", -float(angle), 0, 0, 1)
        
        # Print progress so user knows it's working
        if i % 10 == 0 or i == len(angles) - 1:
            print(f"Generated projection {i:03d}/{len(angles)} at angle {angle}...")

    # Save the scale factors to sino_scales.npy for perfect reconstruction
    scales_path = os.path.join(output_dir, "sino_scales.npy")
    np.save(scales_path, np.array(sino_max_list, dtype=np.float32))

    print(f"All {len(angles)} TIFF frames generated successfully using gVXR!")
    
    # Generate settings.cto for DeepLearningCT compatibility
    settings_content = f"""[Device settings]
mA = 1.000000
kV = 100.000000

[Detector settings]
binning = 0
frames = 1
exp time (ms) = 500.000000
CORunbinned = {det_size_pixels / 2}
pixel size = {det_pixel_mm}
Xmin = 0
Xmax = {det_size_pixels}
Ymin = 0
Ymax = {det_size_pixels}
VC = {det_size_pixels / 2}

[CT scan settings]
projections = {len(angles)}
angle range = 360.000000
CWCCW = FALSE
SDD = {sdd_mm}

[CT reconstruction settings]
SOD = {sod_mm}
SDD = {sdd_mm}
COR = {det_size_pixels / 2}
vertical center = {det_size_pixels / 2}
last angle = 360.000000
bhc = 0.000000
filter strength = 0.000000
projections = {len(angles)}
rows = {det_size_pixels}
columns = {det_size_pixels}
pixel_size (mm) = {det_pixel_mm}
zmax = {det_size_pixels - 1}
zmin = 0
direction = 1
tilt = 0.000000
xminrec = 0
xmaxrec = {det_size_pixels}
yminrec = 0
ymaxrec = {det_size_pixels}
interpolate = FALSE
"""
    cto_path = os.path.join(output_dir, "settings.cto")
    with open(cto_path, "w") as f:
        f.write(settings_content)

    print(f"settings.cto written to {cto_path}")
    
    gvxr.destroyWindow()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate realistic CT projections using gVXR.")
    parser.add_argument("--stl", type=str, default="STL/FINAL30.stl", help="Path to input STL.")
    parser.add_argument("--output_dir", type=str, default="gvxr_projections_tiff", help="Output directory.")
    parser.add_argument("--material", type=str, default="Ti", help="Material of the object (e.g. Ti, Al, Fe).")
    parser.add_argument("--scan-method", type=str, choices=['centered', 'offset', 'auto'], default='auto')
    parser.add_argument("--i0", type=float, default=50000.0, help="Initial photon count.")
    parser.add_argument("--gaussian-std", type=float, default=10.0, help="Gaussian noise standard deviation.")
    args = parser.parse_args()

    run_gvxr_pipeline(args.stl, args.output_dir, args.material, i0=args.i0, gaussian_std=args.gaussian_std, scan_method=args.scan_method)
