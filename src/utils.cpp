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


long getCurrTime(){
    std::chrono::milliseconds time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    );
    return (long) time.count();
}

// //False if we had to end the thread prematurely, else true
// bool join(std::thread *thread,int timeoutSeconds){
//     auto future = std::async(std::launch::async, &std::thread::join, thread);
//     if (future.wait_for(std::chrono::seconds(timeoutSeconds)) //10s max wait
//         == std::future_status::timeout) {
//             //end the offending thread
//             return false;
//     }
//     return true;
// }