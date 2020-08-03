#include <nlohmann/json.hpp>
#include <xtensor/xnpy.hpp>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <cmath>
#include <iomanip>
#include <ctime>
#include <stdio.h>
#include <stdlib.h> /* atoi */
#include <random>
#define BOOST_PYTHON_MAX_ARITY 17
#include <boost/python.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/numpy.hpp>
namespace p = boost::python;
namespace np = boost::python::numpy;

int nx, ny, nz;
bool isinRange(int val, int r);
std::complex<double> ex3d(const std::complex<double> *Ex3d, int z, int y, int x);
bool isin(int a, int b, int c, int d, int l);
std::complex<double> myexp(double xp, double yp, double fm1, double fn1, double fm2, double fn2, double lam);

double sum(const std::complex<double> *Ex3d, int nx, int ny, int nz);

double myrand(double LO, double HI);


std::complex<double> s, Ex3diln1m1, Ex3diln2m2, ar;

double CalcM(
    np::ndarray input_npy,
    double Sx,
    double Sy,
    double dx,
    double dy,
    double sxp,
    double syp,
    double xmin,
    double xmax,
    double ymin,
    double ymax,
    double lmin,
    double lmax,
    double sz,
    int64_t m0,
    int mfold,
    int seed
    )
{
    double dax, day, dl, k0, re, im, V, xp, yp, fm1, fn1, fm2, fn2, lam, xplim, yplim;
    int ixp, iyp, m1, n1, m2, n2, il;
    int m1Mixp, n1Miyp, m2Mixp, n2Miyp;
    std::srand(seed);
    std::default_random_engine generator;
    std::normal_distribution<double> distx(0, sqrt(2.0)*sxp);
    std::normal_distribution<double> disty(0, sqrt(2.0)*syp);
    std::normal_distribution<double> distfm(0, 1.0 / sqrt(2.0)  / Sx);
    std::normal_distribution<double> distfn(0, 1.0 / sqrt(2.0) / Sy);

    auto shape = input_npy.get_shape();
    nx = (int)(shape[2]);
    std::cout << "nx = " << nx << std::endl;
    ny = (int)(shape[1]);
    std::cout << "ny = " << ny << std::endl;
    nz = (int)(shape[0]);
    dax = (xmax-xmin)/(nx-1);
    day = (ymax-ymin)/(ny-1);
    dl = (lmax-lmin)/(nz-1);
    std::cout << "nz = " << nz << std::endl;
    const std::complex<double> *Ex3d = reinterpret_cast<std::complex<double>*>(input_npy.get_data());
    double tot = dl * dax * day * sum(Ex3d, nx, ny, nz);
    std::cout << "tot = " << tot << std::endl;
    V = (lmax-lmin) * pow((xmax-xmin) * (ymax-ymin), 1);
    std::cout << "Starting the loop" << std::endl;
    s = 0;
    int64_t imax = m0*pow(2, mfold)+1;
    ar.real(0.0);
    for (int64_t i=1;i<imax;i++)
    {
        lam = myrand(lmin, lmax);
        k0 = 2 * M_PI / lam;
        double fm1mfm2 = distfm(generator)/k0;
        fm2 = myrand(xmin, xmax);
        fm1 = fm2+fm1mfm2;
        double fn1mfn2 = distfn(generator)/k0;
        fn2 = myrand(ymin, ymax);
        fn1 = fn2+fn1mfn2;
        m1 = int((fm1 - xmin) / dax);
        m2 = int((fm2 - xmin) / dax);
        if (m2 == nx)
        {
            m2 = nx - 1;
        }
        n1 = int((fn1 - ymin) / day);
        n2 = int((fn2 - ymin) / day);
        if (n2 == ny)
        {
            n2 = ny - 1;
        }
        il = int((lam - lmin) / dl);
        if (il == nz)
        {
            il = nz - 1;
        }
        xp = distx(generator);
        yp = disty(generator);
        m1Mixp = int((fm1 - xmin - xp) / dax + 0.5);
        n1Miyp = int((fn1 - ymin - yp) / day + 0.5);
        m2Mixp = int((fm2 - xmin - xp) / dax + 0.5);
        n2Miyp = int((fn2 - ymin - yp) / day + 0.5);
        
        if (isin(m1,n1,m2,n2,il) & isin(m1Mixp, n1Miyp, m2Mixp, n2Miyp, il))
        {
            ar.imag(k0 * dx * (fm1 - fm2) * xp + k0 * dy * (fn1 - fn2) * yp);
            s = s * (double(i - 1) / i) + (1.0 / i) * (2 * M_PI /2 / k0 / k0 / Sx / Sy) * pow(lam, 2) * ex3d(Ex3d, il, n1Miyp, m1Mixp) * conj(ex3d(Ex3d, il, n1, m1) * ex3d(Ex3d, il, n2Miyp, m2Mixp)) * ex3d(Ex3d, il, n2, m2) * exp(-ar);
        }
        if (i == m0)
        {
            std::complex<double> M = pow(tot, 2) / (1.0 / 2.0 / sqrt(M_PI) / sz * V * s);
            std::time_t t = std::time(nullptr);
            std::cout << std::put_time(std::localtime(&t), "%c %Z") << std::endl;
            std::cout << "n points = " << i << ", M = " << M << std::endl;
            m0 = 2 * m0;
        }
    }
    
    std::complex<double> M = pow(tot, 2) / (1.0 / 2.0 / sqrt(M_PI) / sz * V * s);
    return M.real();


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

double sum(const std::complex<double> *Ex3d, int nx, int ny, int nz)
{
    double res = 0;
    #pragma simd
    for (int i = 0; i < nx * ny * nz; i++)
    {
        res += pow(abs(Ex3d[i]), 2);
    }
    return res;
}

double myrand(double LO, double HI){
    return LO + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (HI - LO)));
}


BOOST_PYTHON_MODULE(coherent_modes_cpp)
{
    using namespace boost::python;
    Py_Initialize();
    np::initialize();
    def("CalcM", CalcM);
}