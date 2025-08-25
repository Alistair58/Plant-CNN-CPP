#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <chrono>
#include <future>
#include <thread>
#include <algorithm>
#include <random>

std::vector<std::string> strSplit(std::string str,std::vector<char> delimiters);
uint64_t getCurrTimeMs();
uint64_t getCurrTimeUs();
std::string toLower(std::string s);
inline int max(int a,int b){ return (a>=b)?a:b; }
extern thread_local std::mt19937 localRng;

#endif