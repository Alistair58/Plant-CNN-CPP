#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <chrono>
#include <future>
#include <thread>

std::vector<std::string> strSplit(std::string str,std::vector<char> delimiters);
uint64_t getCurrTimeMs();
uint64_t getCurrTimeUs();
inline int max(int a,int b){ return (a>=b)?a:b; }
// bool join(std::thread *thread,int timeoutSeconds);

#endif