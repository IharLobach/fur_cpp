import coherent_modes_cpp as cm
import numpy as np

params = np.array([    300.96585801243214,
    240.67384278137428,
    1048234.8735736432,
    -487663.40045622655,
    0.00043265807507931146,
    0.00043148229886993364,
    -0.007257142857142857,
    0.007257142857142857,
    -0.00513157492632523,
    0.00513157492632523,
    0.85,
    1.3,
    286275.88162576384,
    0.0,
    100000,
    8,
    1], dtype=np.float64)

print(cm.CalcM(np.load("/mnt/c/Users/lobac_000/OneDrive - Fermi National Accelerator Laboratory/FUR/SRW_SLAC_undulator_spectrum/Ex_3D_with_losses.npy"),
    params))