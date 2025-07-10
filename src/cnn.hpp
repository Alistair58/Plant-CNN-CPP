#ifndef CNN_HPP
#define CNN_HPP

#include <string>
#include "matrices.hpp"
#include "cnnutils.hpp"

class CNN : CnnUtils{
    public:
        //CONSTRUCTORS 
        //Creating a fresh CNN
        CNN(float LR,Dataset *dataset,bool restart);
        //Creating a copy from a template CNN
        CNN(CNN *template,float LR,Dataset *dataset);

        //KEY METHODS 
        std::string forwards(d3 imageInt);
        void backwards(d3 imageInt,std::string answer);
    private:
        //BACKPROPAGATION-RELATED
        void mlpBackwards(d2 dcDzs);
        void convBackwards(d4 dcDxs, int l,bool padding);
        void finalPoolingConvBackwards(d2 dcDzs,d4 dcDxs,bool padding);
        void poolingConvBackwards(d4 dcDxs, int l,bool padding);
};

#endif