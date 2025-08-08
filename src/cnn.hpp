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
        CNN(float LR,Dataset *dataset,bool restart);
        //Creating a copy from an original CNN
        CNN(CNN *original,float LR,Dataset *dataset,bool deepCopy=true);
    
        //KEY METHODS 
        std::string forwards(Tensor& imageInt);
        void backwards(Tensor& imageInt,std::string answer);
    private:
        //BACKPROPAGATION-RELATED
        void mlpBackwards(std::vector<Tensor>& dcDzs);
        void convBackwards(std::vector<Tensor>& dcDxs, int l,bool padding);
        void finalPoolingConvBackwards(std::vector<Tensor>& dcDzs,std::vector<Tensor>& dcDxs,bool padding);
        void poolingConvBackwards(std::vector<Tensor>& dcDxs, int l,bool padding);
};

#endif