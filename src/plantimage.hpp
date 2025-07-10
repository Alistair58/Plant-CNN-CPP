#ifndef PLANTIMAGE_HPP
#define PLANTIMAGE_HPP

#include "matrices.hpp"
#include "globals.hpp"
#include <string>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

class PlantImage {
    public:
        d3 data;
        std::string label = "";
        int index = -1;

        PlantImage::PlantImage(std::string fname, std::string plantName);
        d3 fileToImageArr(std::string fName);
};

#endif PLANTIMAGE_HPP