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

Tensor PlantImage::fileToImageTensor(std::string fName){
    Tensor result;
    #if DEBUG
        uint64_t startTime = getCurrTimeMs();
    #endif

    int width,height,channels;
    unsigned char *img = stbi_load(fName.c_str(),&width,&height,&channels,0);

    if(img==nullptr){
        #if DEBUG
            std::cout << "Could not load \""+fName+"\"" << std::endl;
        #endif
        return result;
    }

    #if DEBUG
        std::cout << "Getting image took "+std::to_string(getCurrTimeMs()-startTime)+"ms" << std::endl;
    #endif
    
    result = Tensor({3,height,width}); //RGB
    //There is no alpha as most images in this dataset are jpeg which don't have an alpha channel
    int pixel;
    #if DEBUG
        uint64_t parsingStart = getCurrTimeMs();
    #endif
    float *resultData = result.getData();
    int gChannel = result.getChildSizes()[0];
    int bChannel = 2*result.getChildSizes()[0];
    for (int y=0;y<height;y++) {
        int rRow = y*result.getChildSizes()[1];
        int gRow = gChannel + rRow;
        int bRow = bChannel + rRow;
        int imgIPart = y*width*channels; // i = (y*width + x)*channels
        for (int x=0;x<width;x++) {
            int imgI  = imgIPart + x*channels;
            resultData[rRow+x] = img[imgI]; //R
            resultData[gRow+x] = img[imgI+1];//G
            resultData[bRow+x] = img[imgI+2]; //B
            //A, if present, is the 3rd offset
        }
    }
    stbi_image_free(img);

    #if DEBUG
        std::cout << "Parsing image took "+std::to_string(getCurrTimeMs()-parsingStart)+"ms" << std::endl;
    #endif

    return result;


}
