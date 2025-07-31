#ifndef CNNUTILS_HPP
#define CNNUTILS_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numbers>
#include "globals.hpp"
#include "tensor.hpp"
#include "dataset.hpp"
#include <algorithm>
#include <random>
#include "json.hpp"

class CNN; //forward declaration needed for compilation of applyGradients

class CnnUtils {
    protected:
        //Things with mutliple layers are stored as vectors as each layer can have different sized tensors
        std::vector<Tensor> kernels; //the kernels are stored [layer][currLayerChannel][prevLayerChannel][y][x] 
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
        void saveMaps();

    public:
        //IMAGE-RELATED
        Tensor parseImg(Tensor& img);
        static void normaliseImg(Tensor& img,std::vector<float> pixelMeans,std::vector<float> pixelStdDevs);
        static Tensor gaussianBlurKernel(int width,int height);
        static Tensor maxPool(Tensor& image,int xStride,int yStride);
        //variable size output
        static Tensor convolution(Tensor& image,Tensor& kernel,int xStride,int yStride,bool padding);
        //fixed size output
        static Tensor convolution(Tensor& image,Tensor& kernel,int xStride,int yStride,int newWidth,int newHeight,bool padding);

        //MATH UTILS
        static std::vector<float> softmax(std::vector<float> inp);
        static inline float sigmoid(float num){
            if (num > 200) return 1;
            if (num < -200) return 0;
            return 1 / (float) (1 + std::exp(-num));
        }
        static inline float relu(float num){
            if (num <= 0) return 0;
            return num;
        }
        static inline float leakyRelu(float num){
            if (num <= 0) return num*0.01f;
            return num;
        }
        static inline bool floatCmp(float x,float y,float epsilon = std::numeric_limits<float>::min()){
            return (x+epsilon>=y && x-epsilon<=y);
        }
        static float normalDistRandom(float mean,float stdDev);
        //Does a modulo but the sign of the output is the sign of y
        //e.g. 
        //floorMod(-5,2) = 1
        //floorMod(5,-2) = -1
        static inline int floorMod(int x, int y) {
            x %= y;
            if (x<0) {
                x += y;
            }
            return x;
        }

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
        void applyGradient(std::vector<Tensor>& values, std::vector<Tensor>& gradient);
};


#endif