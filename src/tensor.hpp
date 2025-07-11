#ifndef tensor.hpp
#define tensor.hpp
#include <vector>
#include <stdlib.h>
#include <string>
#include <stdexcept>

class Tensor{
    float *data;
    std::vector<int> dimens; //e.g. 5x4x6 {5,4,6}
    std::vector<int> childSizes; //e.g. {24,6,1}
    public:
        Tensor(std::vector<int> inputDimens);
        ~Tensor(){ free(data); };
        float operator[](std::vector<int> indices) {
            return data[flattenIndex(indices)];
        }
        std::vector<int> getDimens() { return dimens; }
        float *getData() { return data; }
    
    private:
        size_t flattenIndex(std::vector<int> indices) const;
};
#endif