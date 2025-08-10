#include "utils.hpp"

std::vector<std::string> strSplit(std::string str,std::vector<char> delimiters){
    std::vector<std::string> res;
    int lastI = 0;
    for(int i=0;i<str.length();i++){
        for(char c:delimiters){
            if(c==str[i]){
                //Restrictions for substring to be valid (in bounds and non-zero length)
                if(i>lastI){
                    std::string substring = str.substr(lastI,i-lastI);
                    res.push_back(substring);  
                }
                lastI = i+1;
                break;
            }
        }
    }
    if(lastI!=str.length()){
        std::string finalSubstring = str.substr(lastI,str.length()-lastI);
        res.push_back(finalSubstring);
    }
    return res;
}


uint64_t getCurrTimeMs(){
    std::chrono::milliseconds time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    );
    return (int64_t) time.count();
}

uint64_t getCurrTimeUs(){
    std::chrono::microseconds time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    );
    return (int64_t) time.count();
}

std::string toLower(std::string s){
    std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c){ return std::tolower(c); });
    return s;
}

