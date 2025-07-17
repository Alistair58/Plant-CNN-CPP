#include "plantimage.hpp"

PlantImage::PlantImage(std::string fname, std::string plantName){ //fname can be relative or absolute
    if(fname.length()>2 && fname.substr(0,3)!="C:/"){
        fname = datasetDirPath+fname; 
    }
    this->data = fileToImageArr(fname);
    this->label = plantName;
    std::vector<std::string> fnameSplit = strSplit(fname,{'.','/'});
    if(fnameSplit.size()>1){
        this->index = std::stoi(fnameSplit[fnameSplit.size()-2]);
    }
    
}

Tensor fileToImageArr(std::string fName){
    Tensor result({1,1,1});
    *(result[0]) = -1;
    long startTime = getCurrTime();

    int width,height,channels;
    unsigned char *img = stbi_load(fName.c_str(),&width,&height,&channels,0);

    if(img==nullptr){
        std::cout << "Could not load \""+fName+"\"" << std::endl;
        return result;
    }

    #if DEBUG
        std::cout << "Getting image took "+std::to_string(getCurrTime()-startTime)+"ms" << std::endl;
    #endif
    
    result = Tensor({3,height,width}); //RGB
    //There is no alpha as most images in this dataset are jpeg which don't have an alpha channel
    int pixel;
    long parsingStart = getCurrTime();
    for (int y=0;y<height;y++) {
        for (int x=0;x<width;x++) {
            int i = (y*width + x)*channels;
            *result[{0,y,x}] = img[i]; //R
            *result[{1,y,x}] = img[i+1];//G
            *result[{2,y,x}] = img[i+2]; //B
            //A, if present, is the 3rd offset
        }
    }
    stbi_image_free(img);

    #if DEBUG
        std::cout << "Parsing image took "+std::to_string(getCurrTime()-parsingStart)+"ms" << std::endl;
    #endif
    
    return result;


}
