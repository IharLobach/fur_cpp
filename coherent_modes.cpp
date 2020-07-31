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
// for convenience
using json = nlohmann::json;

int nx, ny, nz;
bool isinRange(int val, int r);
std::complex<double> ex3d(const std::complex<double> *Ex3d, int z, int y, int x);
bool isin(int a, int b, int c, int d, int l);
std::complex<double> myexp(double xp, double yp, double fm1, double fn1, double fm2, double fn2, double lam);

double sum(const std::complex<double> *Ex3d, int nx, int ny, int nz);

double myrand(double LO, double HI);

double Sx, Sy, dx, dy, sxp, syp, dax, day, lmin, lmax, dl, xmin, xmax, ymin, ymax, sz, k0, re, im, V, mfold, xp, yp, fm1, fn1, fm2, fn2, lam, xplim, yplim;
int ixp, iyp, m1, n1, m2, n2, il;
int m1Mixp, n1Miyp, m2Mixp, n2Miyp;
int64_t m0;
std::complex<double> s;

int main(int argc, char* argv[])
{
    int seed = 1;
    if (argc > 2) {
        seed = atoi(argv[2]);
    }
    std::cout << "Random seed = " << seed << std::endl;
    std::ifstream i(argv[1]);
    json j;
    i >> j;
    std::string filepath = j["npy_file_path"];
    std::cout << ".npy file path = " << filepath << std::endl;
    Sx = j["Sx"];
    std::cout << "Sx = " << Sx << std::endl;
    Sy = j["Sy"];
    std::cout << "Sy = " << Sy << std::endl;
    dx = j["dx"];
    std::cout << "dx = " << dx << std::endl;
    dy = j["dy"];
    std::cout << "dy = " << dy << std::endl;
    sxp = j["sxp"];
    std::cout << "sxp = " << sxp << std::endl;
    syp = j["syp"];
    std::cout << "syp = " << syp << std::endl;
    xmin = j["xmin"];
    std::cout << "xmin = " << xmin << std::endl;
    xmax = j["xmax"];
    std::cout << "xmax = " << xmax << std::endl;
    ymin = j["ymin"];
    std::cout << "ymin = " << ymin << std::endl;
    ymax = j["ymax"];
    std::cout << "ymax = " << ymax << std::endl;
    lmin = j["lmin"];
    std::cout << "lmin = " << lmin << std::endl;
    lmax = j["lmax"];
    std::cout << "lmax = " << lmax << std::endl;
    sz = j["sz"];
    std::cout << "sz = " << sz << std::endl;
    m0 = j["m0"];
    std::cout << "m0 = " << m0 << std::endl;
    mfold = j["mfold"];
    std::cout << "mfold = " << mfold << std::endl;

    std::srand(seed);
    std::default_random_engine generator;
    std::normal_distribution<double> distx(0, sqrt(2.0)*sxp);
    std::normal_distribution<double> disty(0, sqrt(2.0)*syp);

    auto data = xt::load_npy<std::complex<double>>(filepath);
    nx = data.shape()[2];
    std::cout << "nx = " << nx << std::endl;
    ny = data.shape()[1];
    std::cout << "ny = " << ny << std::endl;
    nz = data.shape()[0];
    dax = (xmax-xmin)/(nx-1);
    day = (ymax-ymin)/(ny-1);
    dl = (lmax-lmin)/(nz-1);
    xplim = 3*sxp;
    yplim = 3*syp;
    std::cout << "nz = " << nz << std::endl;
    const std::complex<double> *Ex3d = data.data();
    double tot = dl * dax * day * sum(Ex3d, nx, ny, nz);
    std::cout << "tot = " << tot << std::endl;
    V = dl * (nz - 1) * pow(dax * (nx - 1) * day * (ny - 1), 2);
    std::cout << "Starting the loop" << std::endl;
    s = 0;
    int64_t imax = m0*pow(2, mfold)+1;
    for (int64_t i = 1; i<imax; i++)
    { 
        lam = myrand(lmin, lmax);
        fm1 = myrand(xmin, xmax);
        fm2 = myrand(xmin, xmax);
        fn1 = myrand(ymin, ymax);
        fn2 = myrand(ymin, ymax);
        m1 = int((fm1-xmin)/dax);
        if (m1==nx){
            m1 = nx-1;
        }
        m2 = int((fm2-xmin)/dax);
        if (m2==nx){
            m2 = nx-1;
        }
        n1 = int((fn1-ymin)/day);
        if (n1==ny){
            n1 = ny-1;
        }
        n2 = int((fn2-ymin)/day);
        if (n2==ny){
            n2=ny-1;
        }
        xp = distx(generator);
        yp = disty(generator);
        il = int((lam-lmin)/dl);
        if (il==nz){
            il=nz-1;
        }
        m1Mixp = int((fm1-xmin-xp)/dax+0.5);
        n1Miyp = int((fn1-ymin-yp)/day+0.5);
        m2Mixp = int((fm2-xmin-xp)/dax+0.5);
        n2Miyp = int((fn2-ymin-yp)/day+0.5);
        if (isin(m1Mixp, n1Miyp, m2Mixp, n2Miyp, il))
        {
            s = s * (double(i-1)/i) + (1.0/i) * pow(lam,2)*ex3d(Ex3d, il, n1Miyp, m1Mixp) * conj(ex3d(Ex3d, il, n1, m1)* ex3d(Ex3d, il, n2Miyp, m2Mixp)) * ex3d(Ex3d, il, n2, m2) * myexp(xp, yp, fm1, fn1, fm2, fn2, lam);
        }
        if (i == m0)
        {
            std::complex<double> M = pow(tot, 2) / (1.0/2.0/sqrt(M_PI) / sz * V * s);
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

std::complex<double> myexp(double xp, double yp, double fm1, double fn1, double fm2, double fn2, double lam)
{
    k0 = 2 * M_PI / lam;
    re = pow(k0 * Sx * (fm1-fm2), 2) + pow(k0 * Sy * (fn1-fn2), 2);
    im = k0 * dx * (fm1 - fm2) * xp + k0 * dy * (fn1 - fn2) * yp;
    std::complex<double> ar(re, im);
    return exp(-ar);
}

double sum(const std::complex<double> *Ex3d, int nx, int ny, int nz)
{
    double res = 0;
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            for (int k = 0; k < nz; k++)
            {
                res += pow(abs(ex3d(Ex3d, k, j, i)), 2);
            }
        }
    }
    return res;
}

double myrand(double LO, double HI){
    return LO + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (HI - LO)));
}