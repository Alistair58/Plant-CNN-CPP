#include "tensor.hpp"

Tensor::Tensor(const std::vector<int> inputDimens){
    dimens = inputDimens;
    float numElems = 1;
    childSizes.resize(dimens.size());
    for(int i=dimens.size()-1;i>=0;i--){
        childSizes[i] = numElems;
        numElems *= dimens[i];
    }
    totalSize = childSizes[0]*dimens[0];
    //Pointer must be shared as it may be used by sub-tensors
    //initialise values to 0 with float[] constructor
    data = std::shared_ptr<float[]>(new float[totalSize]());
    offset = 0;
}

Tensor::Tensor(const std::vector<int> inputDimens,const std::shared_ptr<float[]> ptr,int pOffset){
    dimens = inputDimens;
    float numElems = 1;
    childSizes.resize(dimens.size());
    for(int i=dimens.size()-1;i>=0;i--){
        childSizes[i] = numElems;
        numElems *= dimens[i];
    }
    data = ptr;
    offset = pOffset;
}

float *Tensor::operator[](const std::vector<int> indices) const{
    if(indices.size()!=dimens.size()){
        throw std::invalid_argument("The length of \"indices\" must match the number of dimensions in the Tensor");
    }
    return (*this)[flattenIndex(indices)];
}

float *Tensor::operator[](int flatIndex) const{
    int realIndex = flatIndex+offset;
    if(realIndex>=totalSize){
        throw std::out_of_range("Index "+std::to_string(realIndex)+" out of bounds for size "+std::to_string(totalSize));
    }
    return (data.get()+realIndex);
}

Tensor Tensor::slice(const std::vector<int> indices) const{
    std::vector<int> subDimens = {dimens.begin()+(dimens.size()-indices.size()),dimens.end()};
    int subOffset = 0;
    for(int i=0;i<(dimens.size()-indices.size());i++){
        subOffset += indices[i]*childSizes[i];
    }
    Tensor subTensor = Tensor(subDimens,data,subOffset);
    return subTensor;
}

void Tensor::operator=(const std::vector<float> vals){
    if(vals.size()!=totalSize){
        throw std::invalid_argument("Length of \"vals\" mismatches size of tensor");
    }
    for(int i=0;i<totalSize;i++){
        data[i+offset] = vals[i];
    }
}

void Tensor::operator=(const Tensor &t){ 
    std::shared_ptr<float[]> tSharedPtr = t.getData();
    if(t.getTotalSize()!=totalSize){
        throw std::invalid_argument("Tensors have mismatched sizes. Assignment cannot take place.");
    }
    for(int i=0;i<totalSize;i++){
        data[i+offset] = tSharedPtr[i];
    }
}

size_t Tensor::flattenIndex(const std::vector<int> indices) const{
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