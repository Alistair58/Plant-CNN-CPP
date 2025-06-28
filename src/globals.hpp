#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include<string>
#include<filesystem>
#include<vector>
#include<exception>

extern std::filesystem::path currDir;
extern std::string datasetDirPath;
extern const std::string ANSI_RED;
extern const std::string ANSI_RESET;
extern const std::string ANSI_GREEN;

class InputMismatchException:public exception;

typedef std::vector<std::vector<float>> d2;
typedef std::vector<std::vector<std::vector<float>>> d3;
typedef std::vector<std::vector<std::vector<std::vector<float>>>> d4;
typedef std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> d5;
#endif