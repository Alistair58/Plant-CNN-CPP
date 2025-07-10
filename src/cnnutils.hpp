#ifndef CNNUTILS_HPP
#define CNNUTILS_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include "globals.hpp"
#include "matrices.hpp"
#include "dataset.hpp"
#include <algorithm>
#include <random>
#include <nlohmann/json.hpp>

class CnnUtils {
    protected:
        d5 kernels; //the kernels are stored [layer][currLayerChannel][prevLayerChannel][y][x] 
    //if there is a bias on a cxkxk kernel, it is at index [c-1][k-1][k]
        d5 kernelsGrad; //This is NOT negative - you must subtract it from the kernels
        d2 activations;
        d3 weights;
        d3 weightsGrad; //Also not negative
        d4 maps; //Note: the input image is included in "maps" for simplicity
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
        d5 loadKernels(bool loadNew);
        d3 loadWeights(bool loadNew);
        //For debugging use
        void saveActivations();

    public:
        //IMAGE-RELATED
        d3 parseImg(d3 img);
        d3 normaliseImg(d3 img,std::vector<float> pixelMeans,std::vector<float> pixelStdDevs);
        d2 gaussianBlurKernel(int width,int height);
        d2 maxPool(d2 image,int xStride,int yStride);
        //variable size output
        d2 convolution(d3 image,d3 kernel,int xStride,int yStride,bool padding);
        //fixed size output
        d2 convolution(d3 image,d3 kernel,int xStride,int yStride,int newWidth,int newHeight,bool padding)

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