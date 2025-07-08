#include "matrices.hpp"


template<typename Matrix>
std::vector<Matrix> newMatrix(const std::vector<int>& dimens){
    if(dimens.size()==1) return std::vector<float>(dimens[0]);
    Matrix child = newMatrix(dimens,1);
    std::vector<Matrix> result = std::vector<Matrix>(dimens[0],child);
    return result;
}

template<typename Matrix>
std::vector<Matrix> newMatrix(const std::vector<int>& dimens, int currIndex){
    if((dimens.size()-1)==currIndex) return std::vector<float>(dimens[0]);
    Matrix child = newMatrix(dimens,currIndex+1);
    std::vector<Matrix> result = std::vector<Matrix>(dimens[currIndex],child);
    return result;
}