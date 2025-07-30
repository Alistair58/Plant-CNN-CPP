#ifndef TENSOR_HPP
#define TENSOR_HPP
#include <vector>
#include <stdlib.h>
#include <string>
#include <stdexcept>
#include <memory>
#include "globals.hpp"

class Tensor{
    std::shared_ptr<float[]> data = nullptr;
    //A class can't have an object of itself and so it must be a pointer
    Tensor *biases = nullptr;
    int offset = 0; //how far this tensor's data is into the shared_ptr
    //OFFSET MUST BE USED FOR EVERY RAW INDEX ACCESS
    std::vector<int> dimens; //e.g. 5x4x6 {5,4,6}
    std::vector<int> childSizes; //e.g. {24,6,1}
    size_t totalSize = 0;
    public:
        //Default constructor, needed for initialising an empty vector with size
        //All the attributes are initialised and so nothing needs to happen
        Tensor(){};
        //Fresh Tensor constructor
        Tensor(const std::vector<int> inputDimens);

        //Rule of 5
        //Biases must only be used for a single tensor
        ~Tensor(){ delete biases; }
        //Copy constructor - needed for deep copy (for biases)
        Tensor(const Tensor& t);
        //Copy assignment operator
        //The values are copied - the memory is not shared
        Tensor& operator=(const Tensor &t);
        //Move constructor
        Tensor(Tensor&& t) noexcept;
        //Move assignment operator
        Tensor& operator=(Tensor&& t) noexcept;

        //Differs from traditional subscript - returns the address
        float *operator[](const std::vector<int> indices) const;
        //Get an address from a flattened index
        float *operator[](int flatIndex) const;
        //Return a subsection of the tensor
        Tensor slice(const std::vector<int> indices) const;
        //Data value assignment by a flat vector
        Tensor& operator=(const std::vector<float> vals);
        
        template <typename dn>
        dn toVector() const;

        std::vector<int> getDimens() const { return dimens; }
        size_t getTotalSize() const { return totalSize; }
        Tensor *getBiases() const { return biases; }
        void setBiases(Tensor *pBiases) { biases = pBiases; }

    private:
        //Sub-Tensor constructor
        Tensor(const std::vector<int> inputDimens,const std::shared_ptr<float[]> ptr,int pOffset);
        size_t flattenIndex(const std::vector<int> indices) const;
        

        
        //Part of toVector
        
        template <typename dn>
        dn buildNestedVector(int depth = 0, int offset = 0) const;
        template<typename dn>
        static constexpr int nestedVectorDepth();

        std::shared_ptr<float[]> getData() const { return data; }
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
    for (int i=0;i<dimens[depth];i++) {
        leaf[i] = data[vectorOffset + i];
    }
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
    return buildNestedVector<dn>(0,0);
}

#endif