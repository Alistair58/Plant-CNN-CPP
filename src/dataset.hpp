#ifndef DATASET_HPP
#define DATASET_HPP

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include "plantimage.hpp"
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "globals.hpp"
namespace fs = std::filesystem;


class Dataset {
    private:
        std::vector<int> indices;
        int size; 
        std::string dirPath;
        std::vector<std::string> fileExtensions; //Includes the dots
        std::vector<std::vector<float>> pixelStats;
        
        void loadPixelStats();
    public:
        std::vector<std::string> plantNames;
        std::unordered_map<std::string,int> plantToIndex;

        Dataset(std::string dirPathInp);
        float getPixelMeans();
        float getPixelStdDevs();
        PlantImage *randomImage(bool test);


};

#endif DATASET_HPP