#ifndef tensor.hpp
#define tensor.hpp
#include <vector>
#include <stdlib.h>
#include <string>
#include <stdexcept>
#include <memory>

class Tensor{
    std::shared_ptr<float[]> data;
    int offset; //how far this tensor's data is into the shared_ptr
    //OFFSET MUST BE USED FOR EVERY INDEX ACCESS
    std::vector<int> dimens; //e.g. 5x4x6 {5,4,6}
    std::vector<int> childSizes; //e.g. {24,6,1}
    size_t totalSize;
    public:
        //Fresh Tensor constructor
        Tensor(const std::vector<int> inputDimens);
        //Sub-Tensor constructor
        Tensor(const std::vector<int> inputDimens,const std::shared_ptr<float[]> ptr,int pOffset);
        //Returns the address
        float *Tensor::operator[](const std::vector<int> indices) const;
        Tensor slice(const std::vector<int> indices) const;
        //Data value assignment by a flat vector
        void operator=(const std::vector<float> vals);
        //Data value assignment by another tensor
        //The values are copied - the memory is not shared
        void operator=(const Tensor &t);

        std::shared_ptr<float[]> getData() const { return data; }
        std::vector<int> getDimens() const { return dimens; }
        size_t getTotalSize() const { return totalSize; }
    private:
        size_t flattenIndex(const std::vector<int> indices) const;
};


#endif