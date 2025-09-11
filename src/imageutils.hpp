#ifndef IMAGEUTILS_HPP
#define IMAGEUTILS_HPP

#include "tensor.hpp"
#include "globals.hpp"
#include <string>
#include <iostream>
#include <cmath>
#include <random>
#include "stb_image.h"
#include "utils.hpp"
#include "stb_image_write.h"

#if PROFILING
    #include "timer.hpp"
#endif

class ImageUtils{
    public:
        Tensor data;
        Tensor fileToImageTensor(std::string fName
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif
        );
        void saveData(std::string fName) const;
        static void rotate(Tensor& inp,float theta
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif
        );
        static void zoom(Tensor& inp,float scaleFactor
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif
        );
        static void toGreyscale(Tensor& inp
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif
        );
        static void gaussianBlur(Tensor& inp,int kernelSize
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif
        );
        static void changeContrast(Tensor& inp,float value
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif
        );
        static void changeBrightness(Tensor& inp,float value
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif
        );
        static void changeSaturation(Tensor& inp,float value
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif
        );
        static void augment(Tensor &inp
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif
        );
    
};




#endif