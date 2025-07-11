#ifndef CNNUTILS_HPP
#define CNNUTILS_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include "globals.hpp"
#include "tensor.hpp"
#include "dataset.hpp"
#include <algorithm>
#include <random>
#include <nlohmann/json.hpp>

class CnnUtils {
    protected:
        Tensor kernels; //the kernels are stored [layer][currLayerChannel][prevLayerChannel][y][x] 
    //if there is a bias on a cxkxk kernel, it is at index [c-1][k-1][k]
        Tensor kernelsGrad; //This is NOT negative - you must subtract it from the kernels
        Tensor activations;
        Tensor weights;
        Tensor weightsGrad; //Also not negative
        Tensor maps; //Note: the input image is included in "maps" for simplicity
        Dataset *d;
        std::vector<float> numNeurons;
        std::vector<float> numMaps; //includes the result of pooling (except final pooling)
        std::vector<float> mapDimens;
        std::vector<float> kernelSizes; //0 represents a pooling layer, the last one is excluded
        std::vector<float> strides; //pooling strides are included
        bool padding;
        bool verbose;
        float LR;

        //UTILS
        void reset();
        Tensor loadKernels(bool loadNew);
        Tensor loadWeights(bool loadNew);
        //For debugging use
        void saveActivations();

    public:
        //IMAGE-RELATED
        Tensor parseImg(Tensor img);
        Tensor normaliseImg(Tensor img,std::vector<float> pixelMeans,std::vector<float> pixelStdDevs);
        Tensor gaussianBlurKernel(int width,int height);
        Tensor maxPool(Tensor image,int xStride,int yStride);
        //variable size output
        Tensor convolution(Tensor image,Tensor kernel,int xStride,int yStride,bool padding);
        //fixed size output
        Tensor convolution(Tensor image,Tensor kernel,int xStride,int yStride,int newWidth,int newHeight,bool padding)

        //MATH UTILS
        std::vector<float> softmax(std::vector<float> inp);
        float sigmoid(float num);
        float relu(float num);
        float leakyRelu(float num);
        inline bool floatCmp(float x,float y){
            return (x+std::numeric_limits<int>::min()>=y && x-std::numeric_limits<int>::min()<=y);
        }
        float normalDistRandom(float mean,float stdDev);

        //UTILS
        void applyGradients();
        void applyGradients(std::vector<CNN*>& cnns);
        void resetKernels();
        void resetWeights();
        void saveWeights();
        void saveKernels();
};

#endif