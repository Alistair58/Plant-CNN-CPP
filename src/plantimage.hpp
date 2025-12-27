#ifndef PLANTIMAGE_HPP
#define PLANTIMAGE_HPP

#include "tensor.hpp"
#include "globals.hpp"
#include "imageutils.hpp"
#include <string>
#include <iostream>
#include "stb_image.h"
#include "utils.hpp"

#if PROFILING
    #include "timer.hpp"
#endif

class PlantImage:public ImageUtils{
    public:
        std::string label = "";
        int index = -1;

        PlantImage() {};
        PlantImage(std::string fname, std::string plantName
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif
        );
        
};

#endif