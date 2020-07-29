#include <xtensor/xnpy.hpp>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <cmath>
#include <iomanip>
#include <ctime>

int nx, ny, nz;
bool isinRange(int val, int r);
std::complex<double> ex3d(const std::complex<double> *Ex3d, int z, int y, int x);
bool isin(int a, int b, int c, int d, int l);
std::complex<double> myexp(int ixp, int iyp, int m1, int n1, int m2, int n2, int il);

double Sx = 300.96585801243214;
double Sy = 240.67384278137428;
double dx = 1048234.8735736432;
double dy = -487663.40045622655;
double sxp = 0.00043265807507931146;
double syp = 0.00043148229886993364;
double dax = 4.9609375e-05;
double day = 4.9614674702039514e-05;
double lmin = 0.85;
double dl = 0.0015000000000000002;
double sz = 30e4;
double axmin;
double aymin;
int ixp, iyp, m1, n1, m2, n2, il;
int m1Mixp, n1Miyp, m2Mixp, n2Miyp;
double k0;
double re, im;
double V;
int64_t m0;
std::complex<double> s;

int main()
{
    auto data = xt::load_npy<std::complex<double>>("/mnt/c/Users/lobac_000/OneDrive - Fermi National Accelerator Laboratory/FUR/SRW_SLAC_undulator_spectrum/Ex_3D.npy");
    std::cout << data.dimension() << std::endl;
    nx = data.shape()[2];
    ny = data.shape()[1];
    nz = data.shape()[0];
    axmin = -dax * (nx - 1) / 2;
    aymin = -day * (ny - 1) / 2;
    const std::complex<double> *Ex3d = data.data();
    V = dl * (nz - 1) * pow(dax * (nx - 1) * day * (ny - 1), 3);
    std::cout << "Starting the loop" << std::endl;
    s = 0;
    m0 = 100000;
    for (int i = 0; true; i++)
    {
        il = std::rand() % nz;
        ixp = std::rand() % nx;
        m1 = std::rand() % nx;
        m2 = std::rand() % nx;
        iyp = std::rand() % ny;
        n1 = std::rand() % ny;
        n2 = std::rand() % ny;
        m1Mixp = m1 - ixp;
        n1Miyp = n1 - iyp;
        m2Mixp = m2 - ixp;
        n2Miyp = n2 - iyp;
        if (isin(m1Mixp, n1Miyp, m2Mixp, n2Miyp, il))
        {
            s += ex3d(Ex3d, il, n1Miyp, m1Mixp) * ex3d(Ex3d, il, n1, m1) * ex3d(Ex3d, il, n2Miyp, m2Mixp) * ex3d(Ex3d, il, n2, m2) * myexp(ixp, iyp, m1, n1, m2, n2, il);
        }
        if (i == m0)
        {
            std::complex<double> M = 1.0 / (sqrt(M_PI) / sz * V / i / 4.0 / M_PI / sxp / syp * s);
            std::time_t t = std::time(nullptr);
            std::cout << std::put_time(std::localtime(&t), "%c %Z") << std::endl;
            std::cout << "n points = " << i << ", M = " << M << std::endl;
            m0 = 2 * m0;
        }
    }

    return 0;

}

std::complex<double> ex3d(const std::complex<double> *Ex3d, int z, int y, int x)
{
    return Ex3d[z * (nx * ny) + y * (nx) + x];
};

bool isinRange(int val, int r)
{
    if ((val < 0) || (val > r - 1))
    {
        return false;
    }
    else
    {
        return true;
    }
}

bool isin(int a, int b, int c, int d, int l)
{
    if (not isinRange(a, nx))
    {
        return false;
    }
    if (not isinRange(b, ny))
    {
        return false;
    }
    if (not isinRange(c, nx))
    {
        return false;
    }
    if (not isinRange(d, ny))
    {
        return false;
    }
    if (not isinRange(l, nz))
    {
        return false;
    }
    return true;
}

std::complex<double> myexp(int ixp, int iyp, int m1, int n1, int m2, int n2, int il)
{
    k0 = 2 * M_PI / (lmin + dl * il);
    re = pow((axmin + dax * ixp) / sxp, 2) / 4 + pow((aymin + day * iyp) / syp, 2) / 4 + pow(k0 * Sx * dax * (m1 - m2), 2) * pow(k0 * Sy * day * (n1 - n2), 2);
    im = k0 * dx * dax * (m1 - m2) * (axmin + dax * ixp) + k0 * dy * day * (n1 - n2) * (aymin + day * iyp);
    std::complex<double> ar(re, im);
    return exp(-ar);
}