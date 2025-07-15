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


inline long getCurrTime(){
    std::chrono::milliseconds time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    );
    return (long) time.count();
}

inline std::vector<std::string> strSplit(std::string str,std::vector<char> delimiters){
    std::vector<std::string> res;
    int lastI = 0;
    for(int i=1;i<str.length();i++){
        for(char c:delimiters){
            if(c==str[i]){
                std::string substring = str.substr(lastI,i-lastI-1);
                res.push_back(substring);
                lastI = i+1;
                break;
            }
        }
    }
    return res;
}

inline bool join(std::thread *thread,int timeoutSeconds){
    auto future = std::async(std::launch::async, &std::thread::join, thread);
    if (future.wait_for(std::chrono::seconds(timeoutSeconds)) //10s max wait
        == std::future_status::timeout) {
            std::terminate();
            return false;
    }
    return true;
}

#endif