#include "plantimage.hpp"

PlantImage::PlantImage(std::string fname, std::string plantName){ //fname can be relative or absolute
    if(fname.length()>2 && fname.substr(0,3)!="C:/"){
        fname = datasetDirPath+fname; 
    }
    this->data = fileToImageTensor(fname);
    this->label = plantName;
    std::vector<std::string> fnameSplit = strSplit(fname,{'.','/'});
    if(fnameSplit.size()>1){
        // C:/.../plantType/123.jpg
        this->index = std::stoi(fnameSplit[fnameSplit.size()-2]);
    }
}


