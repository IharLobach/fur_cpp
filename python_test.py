import coherent_modes_cpp as cm
import numpy as np

paramsSum = np.array(
    [
        -0.007257142857142857,
        0.007257142857142857,
        -0.00513157492632523,
        0.00513157492632523,
        0.85,
        1.3,
        1000,
        8,
        1,
        96.4/0.511,
        1.0
    ], dtype=np.float64
)

tot = cm.CalcSum(paramsSum, np.ones(1, dtype=np.float64), np.array([10.0], dtype=np.float64)[0])
print(tot)

paramsM = np.array([    300.96585801243214,
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
    tot[-1],
    100000,
    6,
    1,
    96.4/0.511,
    1.0], dtype=np.float64)


print(cm.CalcM(paramsM, np.ones(1, dtype=np.float64), np.array([10.0], dtype=np.float64)[0]))