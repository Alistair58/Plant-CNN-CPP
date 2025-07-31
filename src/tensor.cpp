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
    totalSize = childSizes[0]*dimens[0];
    data = ptr;
    offset = pOffset;
}

Tensor::Tensor(const Tensor& t){
    this->offset = 0;
    this->dimens = t.dimens;
    this->childSizes = t.childSizes;
    this->totalSize = t.totalSize;
    //Independent memory to that of the copyee
    this->data = std::shared_ptr<float[]>(new float[totalSize]());
    Tensor *tBiases = t.getBiases();
    if(tBiases!=nullptr){
        this->biases = new Tensor(tBiases->getDimens());
        //Call to copy assignment operator which will copy data
        (*this->biases) = (*tBiases); 
    }
    std::shared_ptr<float[]> tSharedPtr = t.getData();
    for(int i=0;i<totalSize;i++){
        data[i+offset] = tSharedPtr[i+t.offset];
    }
}

Tensor& Tensor::operator=(const Tensor &t){ 
    if(this==&t) return *this; //Self-assignment prevention
    if(this->data==nullptr && this->offset==0){ //If we haven't been initialised i.e. through default constructor
        this->data = std::shared_ptr<float[]>(new float[t.totalSize]());
        this->offset = 0;
        this->dimens = t.dimens;
        this->childSizes = t.childSizes;
        this->totalSize = t.totalSize;
    }
    else{
        //Restrictions are needed to protect sub-tensors from messing up the main tensor
        if(t.dimens.size()!=this->dimens.size()){
            throw std::invalid_argument("Tensor copy assignment can only take place if both operands have the same number of dimensions");
        }
        for(int i=0;i<this->dimens.size();i++){
            if(this->dimens[i]!=t.dimens[i]){
                throw std::invalid_argument("Tensor copy assignment can only take place if both operands have the same dimensions");
            }
        }
    }
    Tensor *tBiases = t.getBiases();
    if(tBiases!=nullptr){
        if(this->biases!=nullptr){
            delete this->biases;
        }
        this->biases = new Tensor(tBiases->getDimens());
        //recursive call which will copy data
        (*this->biases) = (*tBiases); 
    }
    std::shared_ptr<float[]> tSharedPtr = t.getData();
    for(int i=0;i<totalSize;i++){
        data[i+offset] = tSharedPtr[i+t.offset];
    }
    return *this;
}

Tensor::Tensor(Tensor&& t) noexcept{
    this->offset = t.offset;
    this->dimens = std::move(t.dimens);
    this->childSizes = std::move(t.childSizes);
    this->totalSize = t.totalSize;
    this->data = t.data;
    this->biases = t.biases;
    t.data = nullptr;
    t.biases = nullptr; //prevent the biases from being deleted
    t.offset = 0;
    t.totalSize = 0;
}

Tensor& Tensor::operator=(Tensor&& t){
    if(this == &t) return *this; //Self-assignment prevention
    if(this->data==nullptr && this->offset==0){ //If we haven't been initialised i.e. through default constructor
        this->data = std::shared_ptr<float[]>(new float[t.totalSize]());
        this->offset = 0;
        this->dimens = t.dimens;
        this->childSizes = t.childSizes;
        this->totalSize = t.totalSize;
    }
    else{
        //Restrictions are needed to protect sub-tensors from messing up the main tensor
        if(t.dimens.size()!=this->dimens.size()){
            throw std::invalid_argument("Tensor move assignment can only take place if both operands have the same number of dimensions");
        }
        for(int i=0;i<this->dimens.size();i++){
            if(this->dimens[i]!=t.dimens[i]){
                throw std::invalid_argument("Tensor move assignment can only take place if both operands have the same dimensions");
            }
        }
    }
    
    if(this->biases!=nullptr){
        delete this->biases;
    }
    this->biases = t.biases;
    std::shared_ptr<float[]> tSharedPtr = t.getData();
    for(int i=0;i<totalSize;i++){
        data[i+offset] = tSharedPtr[i+t.offset];
    }
    t.data = nullptr;
    t.biases = nullptr; //prevent the biases from being deleted
    t.offset = 0;
    t.totalSize = 0;
    return *this;
}

float *Tensor::operator[](const std::vector<int> indices) const{
    if(indices.size()!=dimens.size()){
        throw std::invalid_argument("The length of \"indices\" must match the number of dimensions in the Tensor");
    }
    return (*this)[flattenIndex(indices)];
}

float *Tensor::operator[](int flatIndex) const{
    if(flatIndex>=totalSize){ //totalSize is just for this sub-tensor and so offset does not need to be taken into account for bounds (assuming sub-tensor is valid)
        throw std::out_of_range("Index "+std::to_string(flatIndex)+" out of bounds for size "+std::to_string(totalSize));
    }
    int realIndex = flatIndex+offset;
    return (data.get()+realIndex);
}

Tensor Tensor::slice(const std::vector<int> indices) const{ 
    if(indices.size()>=dimens.size()){
        throw std::invalid_argument("Too many indices provided for slice");
    }
    std::vector<int> subDimens = {dimens.end()+((int)indices.size()-(int)dimens.size()),dimens.end()};
    int subOffset = 0;
    //For every dimension that the sub-tensor is missing, add it to the offset
    for(int i=0;i<indices.size();i++){
        subOffset += indices[i]*childSizes[i];
    }
    Tensor subTensor = Tensor(subDimens,data,subOffset);
    return subTensor;
}

Tensor& Tensor::operator=(const std::vector<float> vals){
    if(vals.size()!=totalSize){
        throw std::invalid_argument("Length of \"vals\" mismatches size of tensor");
    }
    for(int i=0;i<totalSize;i++){
        data[i+offset] = vals[i];
    }
    return *this;
}



size_t Tensor::flattenIndex(const std::vector<int> indices) const{
    if (indices.size() != dimens.size()) {
        throw std::invalid_argument("Tensor indices provided do not match tensor dimensions");
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



