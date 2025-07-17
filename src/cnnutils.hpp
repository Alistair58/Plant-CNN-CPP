#ifndef CNNUTILS_HPP
#define CNNUTILS_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include "globals.hpp"
#include "tensor.hpp"
#include "dataset.hpp"
#include <algorithm>
#include <random>
#include "json.hpp"
#include "cnn.hpp"

class CnnUtils {
    protected:
        //Things with mutliple layers are stored as vectors as each layer can have different sized tensors
        std::vector<Tensor> kernels; //the kernels are stored [layer][currLayerChannel][prevLayerChannel][y][x] 
    //if there is a bias on a cxkxk kernel, it is at index [c-1][k-1][k]
        std::vector<Tensor> kernelsGrad; //This is NOT negative - you must subtract it from the kernels
        std::vector<Tensor> activations;
        std::vector<Tensor> weights;
        std::vector<Tensor> weightsGrad; //Also not negative
        std::vector<Tensor> maps; //Note: the input image is included in "maps" for simplicity
        Dataset *d;
        std::vector<int> numNeurons;
        std::vector<int> numMaps; //includes the result of pooling (except final pooling)
        std::vector<int> mapDimens;
        std::vector<int> kernelSizes; //0 represents a pooling layer, the last one is excluded
        std::vector<int> strides; //pooling strides are included
        bool padding;
        float LR;

        //UTILS
        void reset();
        std::vector<Tensor> loadKernels(bool loadNew);
        std::vector<Tensor> loadWeights(bool loadNew);
        //For debugging use
        void saveActivations();

    public:
        //IMAGE-RELATED
        Tensor parseImg(Tensor& img);
        Tensor normaliseImg(Tensor& img,std::vector<float> pixelMeans,std::vector<float> pixelStdDevs);
        Tensor gaussianBlurKernel(int width,int height);
        Tensor maxPool(Tensor& image,int xStride,int yStride);
        //variable size output
        Tensor convolution(Tensor& image,Tensor& kernel,int xStride,int yStride,bool padding);
        //fixed size output
        Tensor convolution(Tensor& image,Tensor& kernel,int xStride,int yStride,int newWidth,int newHeight,bool padding);

        //MATH UTILS
        std::vector<float> softmax(std::vector<float> inp);
        float sigmoid(float num);
        float relu(float num);
        float leakyRelu(float num);
        inline bool floatCmp(float x,float y){
            return (x+std::numeric_limits<float>::min()>=y && x-std::numeric_limits<float>::min()<=y);
        }
        float normalDistRandom(float mean,float stdDev);

        //UTILS
        void applyGradients();
        void applyGradients(std::vector<CNN*>& cnns);
        void resetKernels();
        void resetWeights();
        void saveWeights();
        void saveKernels();
        void resetGrad(std::vector<Tensor>& grad);

        //(GET|SET)TERS
        std::vector<int> getNumMaps(){ return numMaps; }
        std::vector<int> getMapDimens(){ return mapDimens; }
    private:
        //INTERNAL UTILS
        void applyGradient(Tensor *values, Tensor *gradient);
};

#endif