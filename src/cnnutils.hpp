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
#include <immintrin.h>
#include "json.hpp"

#if PROFILING
    #include "timer.hpp"
#endif

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
        std::vector<Tensor> paddedMaps; //Reusing padding is better than allocating for every convolutions
        Dataset *d;
        std::vector<int> numNeurons;
        std::vector<int> numMaps; //includes the result of pooling (except final pooling)
        std::vector<int> mapDimens;
        std::vector<int> kernelSizes; //0 represents a pooling layer, the last one is excluded
        std::vector<int> strides; //pooling strides are included
        std::vector<std::unique_ptr<int[]>> maxPoolIndices;
        bool padding;
        float LR;
        float dropoutProb = 0;

        //UTILS
        void reset();
        std::vector<Tensor> loadKernels(bool loadNew
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif 
        );
        std::vector<Tensor> loadWeights(bool loadNew
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif 
        );
        //For debugging use
        void saveActivations();
        void saveMaps();

    public:
        //IMAGE-RELATED
        Tensor parseImg(Tensor& img
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif
        );
        static void normaliseImg(Tensor& img,std::vector<float> pixelMeans,std::vector<float> pixelStdDevs
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif 
        );
        static Tensor gaussianBlurKernel(int width,int height);
        static Tensor maxPool(Tensor& image,int xStride,int yStride);
        Tensor maxPool(Tensor& image,int xStride,int yStride,int *maxPoolIndices);
        //variable size output
        static Tensor convolution(const Tensor& image,Tensor& kernel,int xStride,int yStride,bool padding
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif
        );
        //Saving the padding allocation
        //prePaddingImage doesn't contain the image data, it just needs to be the correct size
        Tensor convolution(const Tensor& image,Tensor& prePaddedImage,Tensor& kernel,const int xStride,const int yStride
        #if PROFILING
            ,Timer *parentTimer
        #endif
        );
        //fixed size output
        static Tensor convolution(Tensor& image,Tensor& kernel,int xStride,int yStride,int newWidth,int newHeight,bool padding
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif
        );

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
        static inline float dotProduct8f(float *X,float *Y);
        static inline float dotProduct8f(__m256 a,__m256 b);
        static inline float horizontalSum(__m256 a);

        //UTILS
        void applyGradients(int batchSize
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif 
        );
        void applyGradients(std::vector<CNN*>& cnns,int batchSize
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif
        );
        void resetKernels(
        #if PROFILING
            Timer *parentTimer = nullptr
        #endif
        );
        void resetWeights(
        #if PROFILING
            Timer *parentTimer = nullptr
        #endif
        );
        void saveWeights(
        #if PROFILING
            Timer *parentTimer = nullptr
        #endif    
        );
        void saveKernels(
        #if PROFILING
            Timer *parentTimer = nullptr
        #endif
        );
        void resetGrad(std::vector<Tensor>& grad);

        //(GET|SET)TERS
        std::vector<int> getNumMaps(){ return numMaps; }
        std::vector<int> getMapDimens(){ return mapDimens; }
    private:
        //INTERNAL UTILS
        void applyGradient(std::vector<Tensor>& values, std::vector<Tensor>& gradient,int batchSize);
};

inline float CnnUtils::dotProduct8f(float *X,float *Y){
    __m256 a = _mm256_loadu_ps(X);       // Load 8 floats
    __m256 b = _mm256_loadu_ps(Y);       // Load 8 floats
    return dotProduct8f(a,b);
}

inline float CnnUtils::dotProduct8f(__m256 a,__m256 b){
    __m256 prod = _mm256_mul_ps(a, b);   // Multiply X[i] * Y[i]
    //Now horizontally sum all 8 floats in prod
    return horizontalSum(prod);
}

inline float CnnUtils::horizontalSum(__m256 a){
    //Horizontally sum all 8 floats in a
    //lower 4 floats
    // {a0,a1,...}
    __m128 low  = _mm256_castps256_ps128(a);          
    //upper 4 floats
    // {a4,a5,...}
    __m128 high = _mm256_extractf128_ps(a, 1);    
    //add lower and upper halves    
    // {a0+a4,a1+a5,...}
    __m128 sum128 = _mm_add_ps(low, high);            
    //Sum the 4 floats in sum128
    //We can't access the elements easily and so we do some shuffling (with a bit of unnecessary parallel additions) 
    //Duplicate the high bits
    // shuf = {a1+a5,a1+a5,a3+a7,a3+a7}
    __m128 shuf = _mm_movehdup_ps(sum128);               
    // sums = {a1+a5+a0+a4,...,a3+a7+a2+a6,...}
    __m128 sums = _mm_add_ps(sum128, shuf);       
    //Move the 2 high floats to the low position
    // sums = {a3+a7+a2+a6,........} 
    shuf = _mm_movehl_ps(shuf, sums);
    //Add lowest floats (a1+a5+a0+a4) + (a3+a7+a2+a6)
    sums = _mm_add_ss(sums, shuf);      
    //Final sum in lowest float
    return _mm_cvtss_f32(sums); 
}


#endif