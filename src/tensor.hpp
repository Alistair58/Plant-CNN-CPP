#ifndef tensor.hpp
#define tensor.hpp
#include <vector>
#include <stdlib.h>
#include <string>
#include <stdexcept>
#include <memory>
#include "globals.hpp"

class Tensor{
    std::shared_ptr<float[]> data;
    //A class can't have an object of itself and so it must be a pointer
    Tensor *biases = nullptr;
    int offset; //how far this tensor's data is into the shared_ptr
    //OFFSET MUST BE USED FOR EVERY RAW INDEX ACCESS
    std::vector<int> dimens; //e.g. 5x4x6 {5,4,6}
    std::vector<int> childSizes; //e.g. {24,6,1}
    size_t totalSize;
    public:
        //Fresh Tensor constructor
        Tensor(const std::vector<int> inputDimens);
        Tensor(const std::vector<int> inputDimens);
        //Sub-Tensor constructor
        Tensor(const std::vector<int> inputDimens,const std::shared_ptr<float[]> ptr,int pOffset);
        //Biases must only be used for a single tensor
        ~Tensor(){ delete biases; }
        //Differs from traditional subscript - returns the address
        float *operator[](const std::vector<int> indices) const;
        //Get an address from a flattened index
        float *operator[](int flatIndex) const;
        //Return a subsection of the tensor
        Tensor slice(const std::vector<int> indices) const;
        //Data value assignment by a flat vector
        void operator=(const std::vector<float> vals);
        //Data value assignment by another tensor
        //The values are copied - the memory is not shared
        void operator=(const Tensor &t);
        template <typename dn>
        dn toVector() const;

        std::shared_ptr<float[]> getData() const { return data; }
        std::vector<int> getDimens() const { return dimens; }
        size_t getTotalSize() const { return totalSize; }
        Tensor *getBiases() const { return biases; }
        void setBiases(Tensor *pBiases) { biases = pBiases; }

    private:
        size_t flattenIndex(const std::vector<int> indices) const;
        template <typename dn>
        dn buildNestedVector(int depth = 0, int offset = 0) const;
        template<typename dn>
        constexpr int nestedVectorDepth();
};


#endif