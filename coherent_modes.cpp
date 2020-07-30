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
// for convenience
using json = nlohmann::json;

int nx, ny, nz;
bool isinRange(int val, int r);
std::complex<double> ex3d(const std::complex<double> *Ex3d, int z, int y, int x);
bool isin(int a, int b, int c, int d, int l);
std::complex<double> myexp(int ixp, int iyp, int m1, int n1, int m2, int n2, int il);

double sum(const std::complex<double> *Ex3d, int nx, int ny, int nz);

double Sx,Sy,dx,dy,sxp,syp,dax,day,lmin,dl,sz,axmin,aymin,k0,re,im,V,mfold;
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
    dax = j["dax"];
    std::cout << "dax = " << dax << std::endl;
    day = j["day"];
    std::cout << "day = " << day << std::endl;
    lmin = j["lmin"];
    std::cout << "lmin = " << lmin << std::endl;
    dl = j["dl"];
    std::cout << "dl = " << dl << std::endl;
    sz = j["sz"];
    std::cout << "sz = " << sz << std::endl;
    m0 = j["m0"];
    std::cout << "m0 = " << m0 << std::endl;
    mfold = j["mfold"];
    std::cout << "mfold = " << mfold << std::endl;

    std::srand(seed);

    auto data = xt::load_npy<std::complex<double>>(filepath);
    nx = data.shape()[2];
    std::cout << "nx = " << nx << std::endl;
    ny = data.shape()[1];
    std::cout << "ny = " << ny << std::endl;
    nz = data.shape()[0];
    std::cout << "nz = " << nz << std::endl;
    axmin = -dax * (nx - 1) / 2.0;
    aymin = -day * (ny - 1) / 2.0;
    const std::complex<double> *Ex3d = data.data();
    double tot = dl * dax * day * sum(Ex3d, nx, ny, nz);
    std::cout << "tot = " << tot << std::endl;
    V = dl * (nz - 1) * pow(dax * (nx - 1) * day * (ny - 1), 2);
    std::cout << "Starting the loop" << std::endl;
    s = 0;
    int64_t imax = m0*pow(2, mfold)+1;
    for (int64_t i = 0; i<imax; i++)
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
            s += pow(lmin+dl*il,2)*ex3d(Ex3d, il, n1Miyp, m1Mixp) * conj(ex3d(Ex3d, il, n1, m1)* ex3d(Ex3d, il, n2Miyp, m2Mixp)) * ex3d(Ex3d, il, n2, m2) * myexp(ixp, iyp, m1, n1, m2, n2, il);
        }
        if (i == m0)
        {
            std::complex<double> M = pow(tot, 2) / (1.0/2.0/sqrt(M_PI) / sz * V / i / 4.0 / M_PI / sxp / syp * s);
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
    re = pow((axmin + dax * ixp) / sxp, 2) / 4 + pow((aymin + day * iyp) / syp, 2) / 4 + pow(k0 * Sx * dax * (m1 - m2), 2) + pow(k0 * Sy * day * (n1 - n2), 2);
    im = k0 * dx * dax * (m1 - m2) * (axmin + dax * ixp) + k0 * dy * day * (n1 - n2) * (aymin + day * iyp);
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