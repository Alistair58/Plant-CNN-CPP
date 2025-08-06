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
        static inline float dotProductUpTo8f(float *X,int lenX,float *Y,int lenY);

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

inline float CnnUtils::dotProduct8f(float *X,float *Y){
    __m256 a = _mm256_loadu_ps(X);       // Load 8 floats
    __m256 b = _mm256_loadu_ps(Y);       // Load 8 floats
    __m256 prod = _mm256_mul_ps(a, b);   // Multiply X[i] * Y[i]
    //Now horizontally sum all 8 floats in prod
    //lower 4 floats
    // {x0*y0,x1*y1,...}
    __m128 low  = _mm256_castps256_ps128(prod);          
    //upper 4 floats
    // {x4*y4,x5*y5,...}
    __m128 high = _mm256_extractf128_ps(prod, 1);    
    //add lower and upper halves    
    // {x0*y0+x4*y4,x1*y1+x5*y5,...}
    __m128 sum128 = _mm_add_ps(low, high);            
    //Sum the 4 floats in sum128
    //let xi*yi+x(i+4)*y(i+4) = ri
    // i.e. r0+r1+r2+r3
    //We can't access the elements easily and so we do some shuffling (with a bit of unnecessary parallel additions) 
    // sum128 = {r0,r1,r2,r3}
    //Duplicate the high bits
    // shuf = {r1,r1,r3,r3}
    __m128 shuf = _mm_movehdup_ps(sum128);               
    // sums = {r0+r1,...,r2+r3,...}
    __m128 sums = _mm_add_ps(sum128, shuf);       
    //Move the 2 high floats to the low position
    // sums = {r2+r3,........} 
    shuf = _mm_movehl_ps(shuf, sums);
    //Add lowest floats (r0+r1) + (r2+r3)
    sums = _mm_add_ss(sums, shuf);      
    //Final sum in lowest float
    return _mm_cvtss_f32(sums); 
}

inline float CnnUtils::dotProductUpTo8f(float *X,int lenX,float *Y,int lenY){
    if(lenX>=8 && lenY>=8) return dotProduct8f(X,Y);
    float A[8] = {0};
    float B[8] = {0};
    int longest = max(lenX,lenY);
    for(int i=0;i<longest;i++){
        if(i<lenX) A[i] = X[i];
        if(i<lenY) B[i] = Y[i];
    }
    return dotProduct8f(A,B);
    
}

#endif