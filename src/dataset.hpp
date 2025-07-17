#ifndef DATASET_HPP
#define DATASET_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <cstdlib>
#include <regex>
#include "plantimage.hpp"
#include "json.hpp"
#include "globals.hpp"
namespace fs = std::filesystem;


class Dataset {
    private:
        std::vector<int> indices;
        int size; 
        std::string dirPath;
        std::vector<std::string> fileExtensions; //Includes the dots
        d2 pixelStats;

        void loadPixelStats();
    public:
        std::vector<std::string> plantNames;
        std::unordered_map<std::string,int> plantToIndex;

        Dataset(std::string dirPathInp);
        std::vector<float> getPixelMeans();
        std::vector<float> getPixelStdDevs();
        PlantImage *randomImage(bool test);
        PlantImage randomImageObj(bool test);
        void compilePixelStats();

};

#endif DATASET_HPP