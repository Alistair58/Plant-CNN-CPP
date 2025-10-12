#include "imageutils.hpp"
#include "cnnutils.hpp"

//Returns empty result if image could not be loaded
Tensor ImageUtils::fileToImageTensor(std::string fName
#if PROFILING
    ,Timer *parentTimer
#endif
){
    Tensor result;
    #if PROFILING
        Timer *gettingImageTimer = nullptr;
        Timer *fileToImageTensorTimer = nullptr;
        if(parentTimer){
            fileToImageTensorTimer = parentTimer->addChildTimer("fileToImageTensor");
            gettingImageTimer = fileToImageTensorTimer->addChildTimer("gettingImage");
        }
    #endif

    int width,height,channels;
    unsigned char *img = stbi_load(fName.c_str(),&width,&height,&channels,0);
    
    if(channels!=3 && channels!=4){
        stbi_image_free(img);
        //Don't throw an error e.g. loading the wrong file extension isn't the end of the world
        return result; //empty - caller checks
    }
    if(img==nullptr){
        #if PROFILING
            if(parentTimer){
                gettingImageTimer->stop();
                fileToImageTensorTimer->stop();
            }
        #endif
        #if DEBUG
            std::cout << "Could not load \""+fName+"\"" << std::endl;
        #endif
        return result;
    }

    #if PROFILING
        Timer *parsingImageTimer = nullptr;
        if(parentTimer){
            gettingImageTimer->stop();
            parsingImageTimer = fileToImageTensorTimer->addChildTimer("parsingImage");
        }
    #endif

    
    result = Tensor({3,height,width}); //RGB
    //There is no alpha as most images in this dataset are jpeg which don't have an alpha channel
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

    #if PROFILING
        if(parentTimer){
            parsingImageTimer->stop();
            fileToImageTensorTimer->stop();
        }
    #endif

    return result;
}

void ImageUtils::saveData(std::string fName) const{
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


//https://www.desmos.com/calculator/hkxcvooamd
void ImageUtils::rotate(Tensor& inp,float theta
#if PROFILING
    ,Timer *parentTimer
#endif
){ //clockwise is positive
    #if PROFILING
        Timer *rotateTimer = nullptr;
        if(parentTimer){
            rotateTimer = parentTimer->addChildTimer("rotate");
        }
    #endif
    std::vector<int> dimens = inp.getDimens();
    std::vector<int> childSizes = inp.getChildSizes();
    const int height = dimens[1];
    const int width = dimens[2];
    Tensor res = Tensor(dimens);
    const int c_y = height/2;
    const int c_x = width/2;
    float* __restrict__ resData = res.getData();
    float *origData = inp.getData();
    const float cosTheta = cos(theta);
    const float sinTheta = sin(theta);
    for(int c=0;c<dimens[0];c++){
        int channel = c*childSizes[0];
        for(int y_p=0;y_p<height;y_p++){
            int row = channel + y_p*width;
            for(int x_p=0;x_p<width;x_p++){
                int x_pc = x_p - c_x;
                int y_pc = y_p - c_y;
                //rotation matrix
                int x = x_pc*cosTheta + y_pc*sinTheta + c_x;
                int y = y_pc*cosTheta - x_pc*sinTheta + c_y;
                if(x>=0 && x<width && y>=0 && y<height){
                    resData[row+x_p] = origData[channel+y*width+x];
                }
            }
        }
    }
    //deep copy
    inp = res;
    #if PROFILING
        if(parentTimer) rotateTimer->stop();
    #endif
}

//1 is no zoom, >1 zoom in, 0< <1 zoom out, <0 flip and zoom
void ImageUtils::zoom(Tensor &inp,float scaleFactor
#if PROFILING
    ,Timer *parentTimer
#endif
){
    #if PROFILING
        Timer *zoomTimer = nullptr;
        if(parentTimer){
            zoomTimer = parentTimer->addChildTimer("zoom");
        }
    #endif
    std::vector<int> dimens = inp.getDimens();
    std::vector<int> childSizes = inp.getChildSizes();
    const int height = dimens[1];
    const int width = dimens[2];
    Tensor res = Tensor(dimens);
    const int c_y = height/2;
    const int c_x = width/2;
    float* __restrict__ resData = res.getData();
    float *origData = inp.getData();
    for(int c=0;c<dimens[0];c++){
        int channel = c*childSizes[0];
        for(int y_p=0;y_p<height;y_p++){
            int row = channel + y_p*width;
            for(int x_p=0;x_p<width;x_p++){
                int x_pc = x_p - c_x;
                int y_pc = y_p - c_y;
                //Inverse scale factor - find where the value of this pixel should come from
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
    #if PROFILING
        if(parentTimer) zoomTimer->stop();
    #endif
}

void ImageUtils::horizontalFlip(Tensor &inp
#if PROFILING
    ,Timer *parentTimer
#endif
    ){
    #if PROFILING
        Timer *horizontalFlipTimer = nullptr;
        if(parentTimer){
            horizontalFlipTimer = parentTimer->addChildTimer("horizontalFlip");
        }
    #endif
    std::vector<int> dimens = inp.getDimens();
    std::vector<int> childSizes = inp.getChildSizes();
    const int height = dimens[1];
    const int width = dimens[2];
    Tensor res = Tensor(dimens);
    float* __restrict__ resData = res.getData();
    float *origData = inp.getData();
    for(int c=0;c<dimens[0];c++){
        int channel = c*childSizes[0];
        for(int y_p=0;y_p<height;y_p++){
            int row = channel + y_p*width;
            for(int x_p=0;x_p<width;x_p++){
                int x = width-x_p-1;
                resData[row+x_p] = origData[channel+y_p*width+x];
            }
        }
    }
    //deep copy
    inp = res;
    #if PROFILING
        if(parentTimer) horizontalFlipTimer->stop();
    #endif
}


void ImageUtils::toGreyscale(Tensor &inp
#if PROFILING
    ,Timer *parentTimer
#endif
){
    #if PROFILING
        Timer *greyscaleTimer = nullptr;
        if(parentTimer){
            greyscaleTimer = parentTimer->addChildTimer("greyscale");
        }
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
    #if PROFILING
        if(parentTimer) greyscaleTimer->stop();
    #endif
}

void ImageUtils::gaussianBlur(Tensor& inp,int kernelSize
#if PROFILING
    ,Timer *parentTimer
#endif
){
    #if PROFILING
        Timer *gaussianBlurTimer = nullptr;
        if(parentTimer){
            gaussianBlurTimer = parentTimer->addChildTimer("gaussianBlur");
        }
    #endif
    std::vector<int> inpDimens = inp.getDimens();
    if(inpDimens.size()!=3){
        throw std::runtime_error("gaussianBlur only works for 3D images");
    }
    Tensor gKernel = CnnUtils::gaussianBlurKernel(kernelSize,kernelSize);
    Tensor gKernel3d = Tensor({1,kernelSize,kernelSize});
    gKernel3d.slice({0}) = gKernel;
    Tensor inp4d = Tensor({inpDimens[0],1,inpDimens[1],inpDimens[2]}); //convolution requires a 3d array (image with multiple channels) 
    //but we only want to process one channel at a time and so we have to store each channel in a separate 3d array
    for(int l=0;l<inpDimens[0];l++){
        //Deep copy
        inp4d.slice({l,0}) = inp.slice({l});
        //Copy-elision
        Tensor sliced = inp4d.slice({l});
        //Deep copy
        inp.slice({l}) = CnnUtils::convolution(sliced,gKernel3d,1,1,true);
    }
    #if PROFILING
        if(parentTimer) gaussianBlurTimer->stop();
    #endif
}

//value = 1 for no change
//value < 1 for darker, value > 1 for brighter
void ImageUtils::changeBrightness(Tensor& inp,float value
#if PROFILING
    ,Timer *parentTimer
#endif
){
    #if PROFILING
        Timer *changeBrightnessTimer = nullptr;
        if(parentTimer){
            changeBrightnessTimer = parentTimer->addChildTimer("changeBrightness");
        }
    #endif
    if(value < 0){
        throw std::runtime_error("value must be positive for changeBrightness");
    }
    float *inpPtr = inp.getData();
    const float *endPtr = inpPtr+inp.getTotalSize();
    for(;inpPtr<endPtr;inpPtr++){
        *inpPtr *= value;
        if(*inpPtr>255) *inpPtr = 255;
    }
    #if PROFILING
        if(parentTimer) changeBrightnessTimer->stop();
    #endif
}

//value = 1 for no change
//value < 1 for less contrast, value > 1 for more contrast
void ImageUtils::changeContrast(Tensor& inp,float value
#if PROFILING
    ,Timer *parentTimer
#endif
){
    #if PROFILING
        Timer *changeContrastTimer = nullptr;
        if(parentTimer){
            changeContrastTimer = parentTimer->addChildTimer("changeContrast");
        }
    #endif
    if(value < 0){
        throw std::runtime_error("value must be positive for changeContrast");
    }
    std::vector<int> childSizes = inp.getChildSizes();
    std::vector<int> dimens = inp.getDimens();
    if(dimens.size()!=3){
        throw std::runtime_error("Image must be 3D to change contrast");
    }
    const int numChannels = dimens[0];
    const int channelSize = childSizes[0];
    std::vector<float> channelMeans(numChannels);
    float *inpData = inp.getData();
    for(int l=0;l<numChannels;l++){
        double sum = 0;
        float *inpPtr = inpData + l*channelSize; 
        float *nextChannel = inpPtr + channelSize;
        for(;inpPtr<nextChannel;inpPtr++){
            sum += *inpPtr;
        }
        channelMeans[l] = sum/channelSize;
    }

    for(int l=0;l<numChannels;l++){
        const float channelMean = channelMeans[l];
        float *inpPtr = inpData + l*channelSize; 
        float *nextChannel = inpPtr + channelSize;
        for(;inpPtr<nextChannel;inpPtr++){
            *inpPtr = (*inpPtr-channelMean)*value + channelMean;
            if(*inpPtr>255) *inpPtr = 255;
            if(*inpPtr<0) *inpPtr = 0;
        }
    }
    #if PROFILING
        if(parentTimer) changeContrastTimer->stop();
    #endif
}

//value = 1 for no change
//value < 1 for less saturation, value > 1 for more saturation
void ImageUtils::changeSaturation(Tensor& inp,float value
#if PROFILING
    ,Timer *parentTimer
#endif
){
    #if PROFILING
        Timer *changeSaturationTimer = nullptr;
        if(parentTimer){
            changeSaturationTimer = parentTimer->addChildTimer("changeSaturation");
        }
    #endif
    if(value < 0){
        throw std::runtime_error("value must be positive for changeSaturation");
    }
    //Calls copy ctor - deep copy
    Tensor greyscale = inp;
    toGreyscale(greyscale);
    float* __restrict__ greyscalePtr = greyscale.getData();
    float *inpPtr = inp.getData();
    size_t size = inp.getTotalSize();
    float *endPtr = inpPtr+size;
    const float valueComplement = 1-value;
    for(;inpPtr<endPtr;inpPtr++,greyscalePtr++){
        *inpPtr = *inpPtr*value + *greyscalePtr*valueComplement;
        if(*inpPtr>255) *inpPtr = 255;
        if(*inpPtr<0) *inpPtr = 0;
    }
    #if PROFILING
        if(parentTimer) changeSaturationTimer->stop();
    #endif
}

void ImageUtils::augment(Tensor &inp
#if PROFILING
    ,Timer *parentTimer
#endif
){
    #if PROFILING
        Timer *augmentTimer = nullptr;
        if(parentTimer) augmentTimer = parentTimer->addChildTimer("augment");
    #endif
    bool greyscaled = false;
    std::uniform_real_distribution<double> augmentOrNot(0, 1);
    double prob = augmentOrNot(localRng);
    if(prob<0.25){ //Zoom in on 1 in 4
        //If you zoom in somewhere, other than the centre, you might miss the plant
        std::uniform_real_distribution<double> scaleFactorDist(1.25,2);
        ImageUtils::zoom(inp,scaleFactorDist(localRng)
        #if PROFILING
            ,augmentTimer
        #endif
        );
    }
    prob = augmentOrNot(localRng);
    if(prob<0.25){ //Rotate 1 in 4
        std::uniform_real_distribution<double> angleDist(-std::numbers::pi/4,std::numbers::pi/4);
        ImageUtils::rotate(inp,angleDist(localRng)
        #if PROFILING
            ,augmentTimer
        #endif
        );
    }
    prob = augmentOrNot(localRng);
    if(prob<0.2){ //Flip 1 in 5
        ImageUtils::horizontalFlip(inp
        #if PROFILING
            ,augmentTimer
        #endif
        );
    }
    prob = augmentOrNot(localRng);
    if(prob<0.05){ //Greyscale 1 in 20
        ImageUtils::toGreyscale(inp
        #if PROFILING
            ,augmentTimer
        #endif
        );
        greyscaled = true;
    }
   
    prob = augmentOrNot(localRng);
    if(prob<0.0625){ //Blur 1 in 16
        gaussianBlur(inp,5 //5x5 kernel
        #if PROFILING
            ,augmentTimer
        #endif
        );  
    }
    prob = augmentOrNot(localRng);
    if(prob<0.2){ //Change brightness on 1 in 5
        std::uniform_real_distribution<float> brightnessDist(0.5,1.75); 
        float brightnessFactor = brightnessDist(localRng);
        changeBrightness(inp,brightnessFactor
        #if PROFILING
            ,augmentTimer
        #endif
        );
    }

    //Color-related 
    //Do not apply to greyscale
    if(greyscaled){
        #if PROFILING
            if(parentTimer) augmentTimer->stop();
        #endif
        return;
    }
    prob = augmentOrNot(localRng);
    if(prob<0.2){ //Change contrast on 1 in 5
        std::uniform_real_distribution<float> contrastDist(0.5,2); 
        float contrastFactor = contrastDist(localRng);
        changeContrast(inp,contrastFactor
        #if PROFILING
            ,augmentTimer
        #endif
        );
    }
    prob = augmentOrNot(localRng);
    if(prob<0.2){ //Change saturation on 1 in 5
        std::uniform_real_distribution<float> saturationDist(0.5,2); 
        float saturationFactor = saturationDist(localRng);
        changeSaturation(inp,saturationFactor
        #if PROFILING
            ,augmentTimer
        #endif
        );
    }
    #if PROFILING
        if(parentTimer) augmentTimer->stop();
    #endif
}