#include <cnpy.h>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

int main()
{
    cnpy::NpyArray arr = cnpy::npy_load("/mnt/c/Users/lobac_000/OneDrive - Fermi National Accelerator Laboratory/FUR/SRW_SLAC_undulator_spectrum/Ex_3D.npy");
    std::complex<double> *loaded_data = arr.data<std::complex<double>>();

    std::cout << "Element 1,1,1 = " << loaded_data[1,1,1].real() << std::endl;
    std::cout << "Word_size = " << arr.word_size << std::endl;
    std::cout << "Shape = " << arr.shape[0] << ", " << arr.shape[1] << ", " << arr.shape[2] << std::endl;

}