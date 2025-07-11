#include "tensor.hpp"

Tensor::Tensor(std::vector<int> inputDimens){
    dimens = inputDimens;
    float numElems = 1;
    for(int i=dimens.size()-1;i>=0;i--){
        childSizes[i] = numElems;
        numElems *= dimens[i];
        
    }
    data = (float*) calloc(numElems,sizeof(float));
}

size_t Tensor::flattenIndex(std::vector<int> indices) const{
    if (indices.size() != dimens.size()) {
        throw std::invalid_argument("Tensor indices provided do not match tensor shape");
    }

    size_t index = 0;
    for (size_t i=0;i<dimens.size();i++) {
        if (indices[i]<0 || indices[i]>=dimens[i]){
            throw std::out_of_range(
                "Tensor index out of bounds. Index "+std::to_string(indices[i])+
                " does not exist for size "+std::to_string(dimens[i])+"."
            );
        }
        index += indices[i]*childSizes[i];
    }
    return index;
}