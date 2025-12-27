#include "plantimage.hpp"

PlantImage::PlantImage(std::string fname, std::string plantName
#if PROFILING
    ,Timer *parentTimer
#endif
){ //fname can be relative or absolute
    #ifdef __linux__ 
        if(fname.length()>0 && fname[0]!='/'){
            fname = datasetDirPath+fname; 
        }
    #elif _WIN32
        if(fname.length()>2 && fname.substr(0,3)!="C:/"){
            fname = datasetDirPath+fname; 
        }   
    #endif
    
    this->data = fileToImageTensor(fname
    #if PROFILING
            ,parentTimer
    #endif
);
    this->label = plantName;
    std::vector<std::string> fnameSplit = strSplit(fname,{'.','/'});
    if(fnameSplit.size()>1){
        // C:/.../plantType/123.jpg
        this->index = std::stoi(fnameSplit[fnameSplit.size()-2]);
    }
}


