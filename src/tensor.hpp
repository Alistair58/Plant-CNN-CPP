#ifndef TENSOR_HPP
#define TENSOR_HPP
#include <vector>
#include <stdlib.h>
#include <string>
#include <stdexcept>
#include <memory>
#include <cstring>
#include "globals.hpp"


class Tensor{
    std::shared_ptr<float[]> data = nullptr;
    //A class can't have an object of itself and so it must be a pointer
    //Biases aren't shared very often but could be
    std::shared_ptr<Tensor> biases = nullptr;
    int offset = 0; //how far this tensor's data is into the shared_ptr
    std::vector<int> dimens; //e.g. 5x4x6 {5,4,6}
    std::vector<int> childSizes; //e.g. {24,6,1}
    size_t totalSize = 0;
    public:
        //Default constructor, needed for initialising an empty vector with size
        //All the attributes are initialised and so nothing needs to happen
        Tensor(){};
        //Fresh Tensor constructor
        Tensor(const std::vector<int>& inputDimens);

        //not a Rule of 5 as we don't have any raw ptrs and hence don't need a destructor
        //Copy constructor - needed for deep copy (for biases)
        Tensor(const Tensor& t);
        //Copy assignment operator
        //The values are copied - the memory is not shared
        Tensor& operator=(const Tensor &t);
        //Move constructor
        Tensor(Tensor&& t) noexcept;
        //Move assignment operator
        Tensor& operator=(Tensor&& t);

        void shallowCopy(Tensor& src);

        //Differs from traditional subscript - returns the address
        inline float *operator[](const std::vector<int> indices) const{
            if(indices.size()!=dimens.size()){
                throw std::invalid_argument("The length of \"indices\" must match the number of dimensions in the Tensor");
            }
            return (*this)[flattenIndex(indices)];
        }
        //Get an address from a flattened index
        inline float *operator[](size_t flatIndex) const{
            if(flatIndex>=totalSize){ //totalSize is just for this sub-tensor and so offset does not need to be taken into account for bounds (assuming sub-tensor is valid)
                throw std::out_of_range("Index "+std::to_string(flatIndex)+" out of bounds for size "+std::to_string(totalSize));
            }
            size_t realIndex = flatIndex+ (size_t)offset;
            return (data.get()+realIndex);
        }
        //Return a subsection of the tensor
        Tensor slice(const std::vector<int>& indices) const;
        //Return a subsection of the tensor with some biases included
        Tensor slice(const std::vector<int>& indices,const std::vector<int>& biasesIndices) const;
        //Data value assignment by a flat vector
        Tensor& operator=(const std::vector<float>& vals);
        
        template <typename dn>
        dn toVector() const;

        std::vector<int> getDimens() const { return dimens; }
        size_t getTotalSize() const { return totalSize; }
        Tensor *getBiases() const { return biases==nullptr ? nullptr : biases.get(); }
        std::vector<int> getChildSizes() const { return childSizes; }
        int getOffset() const { return offset; }
        void setBiases(Tensor& pBiases) { 
            //Deep copy ctor
            //Old biases ptr is automatically deleted
            biases = std::make_shared<Tensor>(pBiases);
        }
        float *getData() const { return data.get()+offset; }

    private:
        //Sub-Tensor constructor
        Tensor(const std::vector<int>& inputDimens,const std::shared_ptr<float[]> ptr,int pOffset);
        inline size_t flattenIndex(const std::vector<int>& indices) const{
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
        //Part of toVector
        template <typename dn>
        dn buildNestedVector(int depth = 0, int offset = 0) const;
        template<typename dn>
        static constexpr int nestedVectorDepth();

};


//Template methods need to be in the header file
template <typename dn>
dn Tensor::buildNestedVector(int depth, int vectorOffset) const{
    dn vec(dimens[depth]);
    for (int i=0;i<dimens[depth];i++) {
        int newOffset = vectorOffset + i * childSizes[depth];
        //get the value_type of our current template type
        //e.g. value_type of d3 is d2
        vec[i] = buildNestedVector<typename dn::value_type>(depth+1,newOffset);
    }
    return vec;
}

//Base case
template<>
inline d1 Tensor::buildNestedVector<d1>(int depth,int vectorOffset) const{
    d1 leaf(dimens[depth]);
    std::memcpy(
        leaf.data(),
        data.get()+vectorOffset,
        sizeof(float)*dimens[depth]
    );
    return leaf;
}

//Basest case - shouldn't ever be reached but compiler wants itS
template<>
inline float Tensor::buildNestedVector<float>(int depth, int vectorOffset) const {
    return data[vectorOffset];
}

template<typename dn>
//constexpr means that the value of this can be calculated at compile time
//e.g. nestedVectorDepth<d3>() will always return the same value
constexpr int Tensor::nestedVectorDepth() {
    if constexpr (std::is_same<dn, float>::value) {
        return 0;
    } else {
        return 1 + nestedVectorDepth<typename dn::value_type>();
    }
}

template <typename dn>
dn Tensor::toVector() const{
    constexpr int requestedDepth = nestedVectorDepth<dn>();
    int tensorDepth = (int) dimens.size();
    if (requestedDepth != tensorDepth) {
        throw std::invalid_argument(
            "Requested vector depth (" + std::to_string(requestedDepth) +
            ") does not match tensor dimensions (" + std::to_string(tensorDepth) + ")."
        );
    }
    return buildNestedVector<dn>(0,this->offset);
}

#endif