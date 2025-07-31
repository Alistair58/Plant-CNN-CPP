#ifndef PLANTIMAGE_HPP
#define PLANTIMAGE_HPP

#include "tensor.hpp"
#include "globals.hpp"
#include <string>
#include <iostream>
#include "stb_image.h"
#include "utils.hpp"

class PlantImage {
    public:
        Tensor data;
        std::string label = "";
        int index = -1;

        PlantImage() {};
        PlantImage(std::string fname, std::string plantName);
        Tensor fileToImageTensor(std::string fName);
};

#endif