#ifndef CNN_HPP
#define CNN_HPP

#include <string>
#include "tensor.hpp"
#include "cnnutils.hpp"

class CNN : public CnnUtils{
    public:
        //CONSTRUCTORS 
        //Creating a fresh CNN
        CNN(float LR,Dataset *dataset,bool restart);
        //Creating a copy from an original CNN
        CNN(CNN *original,float LR,Dataset *dataset);
    
        //KEY METHODS 
        std::string forwards(Tensor imageInt);
        void backwards(Tensor imageInt,std::string answer);
    private:
        //BACKPROPAGATION-RELATED
        void mlpBackwards(Tensor dcDzs);
        void convBackwards(Tensor dcDxs, int l,bool padding);
        void finalPoolingConvBackwards(Tensor dcDzs,Tensor dcDxs,bool padding);
        void poolingConvBackwards(Tensor dcDxs, int l,bool padding);
};

#endif