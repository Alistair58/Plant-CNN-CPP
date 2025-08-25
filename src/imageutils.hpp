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

class ImageUtils{
    public:
        Tensor data;
        Tensor fileToImageTensor(std::string fName);
        static void rotate(Tensor& inp,float theta);
        static void zoom(Tensor& inp,float scaleFactor);
        static void toGreyscale(Tensor& inp);
        void saveData(std::string fName);
        static void augment(Tensor &inp);
    
};




#endif