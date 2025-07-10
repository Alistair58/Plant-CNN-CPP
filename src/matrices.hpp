#ifndef MATRICES_HPP
#define MATRICES_HPP
#include <vector>
typedef std::vector<float> d1;
typedef std::vector<std::vector<float>> d2;
typedef std::vector<std::vector<std::vector<float>>> d3;
typedef std::vector<std::vector<std::vector<std::vector<float>>>> d4;
typedef std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> d5;
template<typename Matrix>
std::vector<Matrix> newMatrix(const std::vector<int>& dimens);
template<typename Matrix>
std::vector<Matrix> newMatrix(const std::vector<int>& dimens, int currIndex);
#endif