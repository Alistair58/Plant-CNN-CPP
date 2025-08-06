#ifndef GLOBALS_HPP
#define GLOBALS_HPP


#include <string>
#include <filesystem>
#include <vector>
#include <exception>
#include <chrono>
#include <thread>
#include <future>

extern std::string currDir;
extern std::string datasetDirPath;
extern const std::string ANSI_RED;
extern const std::string ANSI_RESET;
extern const std::string ANSI_GREEN;

typedef std::vector<float> d1;
typedef std::vector<d1> d2; 
typedef std::vector<d2> d3; 
typedef std::vector<d3> d4; 
typedef std::vector<d4> d5; 



#endif