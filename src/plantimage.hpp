#ifndef PLANTIMAGE_HPP
#define PLANTIMAGE_HPP

#include "tensor.hpp"
#include "globals.hpp"
#include "imageutils.hpp"
#include <string>
#include <iostream>
#include "stb_image.h"
#include "utils.hpp"

class PlantImage:public ImageUtils{
    public:
        std::string label = "";
        int index = -1;

        PlantImage() {};
        PlantImage(std::string fname, std::string plantName);
        
};

#endif