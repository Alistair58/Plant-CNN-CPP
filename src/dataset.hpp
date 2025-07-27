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
        //The image numbers which mark the beginning of a new type of plant
        std::vector<int> indices;
        int size; 
        std::string dirPath;
        std::vector<std::string> fileExtensions; //Includes the dots
        d2 pixelStats;
        float trainTestSplit = 0.8f;

        void loadPixelStats();
    public:
        //In the same order as the output neurons of CNN
        std::vector<std::string> plantNames;
        //Plant name to the index of CNN output neuron
        std::unordered_map<std::string,int> plantToIndex;

        Dataset(std::string dirPathIn,float trainTestSplitRatio);
        std::vector<float> getPixelMeans();
        std::vector<float> getPixelStdDevs();
        PlantImage *randomImage(bool test);
        PlantImage randomImageObj(bool test);
        void compilePixelStats();

};

#endif