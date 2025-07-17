#ifndef PLANTIMAGE_HPP
#define PLANTIMAGE_HPP

#include "tensor.hpp"
#include "globals.hpp"
#include <string>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

class PlantImage {
    public:
        Tensor data = Tensor({0});
        std::string label = "";
        int index = -1;

        PlantImage() {};
        PlantImage(std::string fname, std::string plantName);
        Tensor fileToImageArr(std::string fName);
};

#endif