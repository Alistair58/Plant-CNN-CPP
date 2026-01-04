#ifndef CNN_HPP
#define CNN_HPP

#include <string>
#include <unordered_map>
#include "tensor.hpp"
#include "cnnutils.hpp"

class CNN : public CnnUtils{
    public:
        //CONSTRUCTORS 
        //Creating a fresh CNN
        CNN(
            float LR,
            Dataset *dataset,
            bool restart,
            float dropoutProbability,
            std::vector<int> numNeurons,
            std::vector<dimens> mapDimens,
            std::vector<std::pair<int,int>> kernelSizes,
            std::vector<std::pair<int,int>> strides,
            bool padding 
        );
        //Creating a copy from an original CNN
        CNN(CNN *original,float LR,Dataset *dataset,bool deepCopyWeights=true);
    
        //KEY METHODS 
        std::string forwards(Tensor& imageInt,bool training
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif 
        );
        void backwards(Tensor& imageInt,std::string answer
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif
        );
    private:
        //FORWARDS-RELATED
        void finalPooling(
        #if PROFILING
            Timer *parentTimer = nullptr
        #endif
        );
        void mlpForwards(bool training
        #if PROFILING
            ,Timer *parentTimer = nullptr
        #endif
        );
        //BACKPROPAGATION-RELATED
        void mlpBackwards(std::vector<Tensor>& dcDzs);
        void convBackwards(std::vector<Tensor>& dcDxs, int l,bool padding
        #if PROFILING
            ,Timer *parentTimer
        #endif
        );
        void finalPoolingConvBackwards(std::vector<Tensor>& dcDzs,std::vector<Tensor>& dcDxs,bool padding);
        void poolingConvBackwards(std::vector<Tensor>& dcDxs, int l,bool padding);
};

#endif