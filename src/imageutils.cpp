#include "imageutils.hpp"

Tensor ImageUtils::fileToImageTensor(std::string fName){
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

//https://www.desmos.com/calculator/hkxcvooamd
void ImageUtils::rotate(Tensor& inp,float theta){ //clockwise is positive
    #if DEBUG
        uint64_t startTime = getCurrTimeMs();
    #endif
    std::vector<int> dimens = inp.getDimens();
    std::vector<int> childSizes = inp.getChildSizes();
    const int height = dimens[1];
    const int width = dimens[2];
    Tensor res = Tensor(dimens);
    int c_y = height/2;
    int c_x = width/2;
    float *resData = res.getData();
    float *origData = inp.getData();
    for(int c=0;c<dimens[0];c++){
        int channel = c*childSizes[0];
        for(int y_p=0;y_p<height;y_p++){
            int row = channel + y_p*width;
            for(int x_p=0;x_p<width;x_p++){
                int x_pc = x_p - c_x;
                int y_pc = y_p - c_y;
                //rotation matrix
                int x = x_pc*cos(theta) + y_pc*sin(theta) + c_x;
                int y = y_pc*cos(theta) - x_pc*sin(theta) + c_y;
                if(x>=0 && x<width && y>=0 && y<height){
                    resData[row+x_p] = origData[channel+y*width+x];
                }
            }
        }
    }
    //deep copy
    inp = res;
    #if DEBUG
        std::cout << "rotate took " << (getCurrTimeMs()-startTime) << "ms" << std::endl;
    #endif
}

//1 is no zoom, >1 zoom in, 0< <1 zoom out, <0 flip and zoom
void ImageUtils::zoom(Tensor &inp,float scaleFactor){
    #if DEBUG
        uint64_t startTime = getCurrTimeMs();
    #endif
    std::vector<int> dimens = inp.getDimens();
    std::vector<int> childSizes = inp.getChildSizes();
    const int height = dimens[1];
    const int width = dimens[2];
    Tensor res = Tensor(dimens);
    int c_y = height/2;
    int c_x = width/2;
    float *resData = res.getData();
    float *origData = inp.getData();
    for(int c=0;c<dimens[0];c++){
        int channel = c*childSizes[0];
        for(int y_p=0;y_p<height;y_p++){
            int row = channel + y_p*width;
            for(int x_p=0;x_p<width;x_p++){
                int x_pc = x_p - c_x;
                int y_pc = y_p - c_y;
                //rotation matrix
                int x = x_pc/scaleFactor + c_x;
                int y = y_pc/scaleFactor + c_y;
                if(x>=0 && x<width && y>=0 && y<height){
                    resData[row+x_p] = origData[channel+y*width+x];
                }
            }
        }
    }
    //deep copy
    inp = res;
    #if DEBUG
        std::cout << "zoom took " << (getCurrTimeMs()-startTime) << "ms" << std::endl;
    #endif
}

void ImageUtils::toGreyscale(Tensor &inp){
    #if DEBUG
        uint64_t startTime = getCurrTimeMs();
    #endif
    std::vector<int> dimens = inp.getDimens();
    std::vector<int> childSizes = inp.getChildSizes();
    const int height = dimens[1];
    const int width = dimens[2];
    Tensor res = Tensor(dimens);
    float *resData = res.getData();
    float *origData = inp.getData();
    for(int y=0;y<height;y++){
        int row = y*width;
        for(int x=0;x<width;x++){
            float greyscaleVal = 0;
            for(int c=0;c<dimens[0];c++){
                int channel = c*childSizes[0];
                greyscaleVal += origData[channel+row+x];
            }
            greyscaleVal /= dimens[0];
            for(int c=0;c<dimens[0];c++){
                int channel = c*childSizes[0];
                resData[channel+row+x] = greyscaleVal;
            }
        }
    }
    //deep copy
    inp = res;
    #if DEBUG
        std::cout << "toGreyscale took " << (getCurrTimeMs()-startTime) << "ms" << std::endl;
    #endif
}

void ImageUtils::saveData(std::string fName){
    if(fName.length()>2 && fName.substr(0,3)!="C:/"){
        fName = currDir+"\\"+fName; 
    }
    std::vector<int> dimens = data.getDimens();
    if(dimens.size()!=3 || (dimens[0]!=3 && dimens[0]!=4)){
        throw std::invalid_argument("saveData can only be used on RGB or ARGB images");
    }
    int height = dimens[1];
    int width = dimens[2];
    std::vector<int> childSizes = data.getChildSizes();
    unsigned char *outputData = new unsigned char[height*width*3];
    float *inputData = data.getData();
    int channel1 = childSizes[0];
    int channel2 = 2*childSizes[0];
    for(int y=0;y<height;y++){
        int channel0Row = y*width;
        int channel1Row = channel1 + channel0Row;
        int channel2Row = channel2 + channel0Row;
        for(int x=0;x<width;x++){
            int i = (y*width + x)*3;
            outputData[i] = (unsigned char) inputData[channel0Row+x];
            outputData[i+1] = (unsigned char) inputData[channel1Row+x];
            outputData[i+2] = (unsigned char) inputData[channel2Row+x];
        }
    }
    if(!stbi_write_jpg((fName).c_str(),width,height,3,outputData,width*height*3)){
        std::cerr << "Could not save image\n";
    }
    else {
        std::cout << "Saved "<< fName << std::endl;
    }
    delete[] outputData;
}

void ImageUtils::augment(Tensor &inp){
    std::uniform_real_distribution<double> augmentOrNot(0, 1);
    double prob = augmentOrNot(localRng);
    if(prob<0.2){ //zoom in on 1 in 5
        //If you zoom in somewhere, other than the centre, you might miss the plant
        std::uniform_real_distribution<double> scaleFactorDist(1,1.75);
        ImageUtils::zoom(inp,scaleFactorDist(localRng));
    }
    prob = augmentOrNot(localRng);
    if(prob<0.2){ //rotate 1 in 5
        std::uniform_real_distribution<double> angleDist(-std::numbers::pi/6,std::numbers::pi/6);
        ImageUtils::rotate(inp,angleDist(localRng));
    }
    prob = augmentOrNot(localRng);
    if(prob<0.05){ //greyscale 1 in 20
        ImageUtils::toGreyscale(inp);
    }
}