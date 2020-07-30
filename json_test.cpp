#include <nlohmann/json.hpp>
#include <fstream>
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

int main(int argc, char *argv[])
{
    int seed = 1;
    seed = atoi(argv[1]);
    std::cout << "seed = " << seed << std::endl;
    std::ifstream i("conf.json");
    json j;
    i >> j;
    std::string filepath = j["npy_file_path"];
    std::cout << "File path = " << filepath << std::endl;
    double Sx = j["Sx"];
    std::cout << "Sx = " << Sx << std::endl;
    return 0;
}