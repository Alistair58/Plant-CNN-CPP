#include "cnnutils.hpp"

//----------------------------------------------------
//IMAGE-RELATED
Tensor CnnUtils::parseImg(Tensor img){
    //The produced images may have a slight black border around them
    //Keeping a constant stride doesn't stretch the image 
    //but as it is an integer means that it will create a border
    //e.g. a 258x258 image would be given a stride length of 2 and so would only have 128 pixels in the remaining image
    std::vector<int> imgDimens = img.getDimens();
    if(imgDimens.size()!=3){
        throw std::invalid_argument("Image must have 3 dimensions for parseImg");
    }
    int channels = imgDimens[0];
    int imHeight = imgDimens[1];
    int imWidth = imgDimens[2];
    //ceil so that we take too large steps and so we take <=mapDimens[0] steps
    //If we floor it, we go out of the mapDimens[0]xmapDimens[0] bounds (as we aren't striding far enough)
    int xStride = (int) std::ceil((float)imWidth/mapDimens[0]); //Reducing size to mapDimens[0] x mapDimens[0] via a Gaussian blur
    int yStride = (int) std::ceil((float)imHeight/mapDimens[0]); 
    Tensor gKernel = gaussianBlurKernel(xStride,yStride);
    Tensor gKernel3d = Tensor({1,xStride,yStride});
    gKernel3d.slice({0}) = gKernel;
    Tensor result = Tensor({channels,imHeight,imWidth});
    Tensor img4d = Tensor({3,1,imHeight,imWidth}); //convolution requires a 3d array (image with multiple channels) 
    //but we only want to process one channel at a time and so we have to store each channel in a separate 3d array
    for(int l=0;l<channels;l++){
        img4d.slice({l,0}) = img.slice({l});
        result.slice({l}) = convolution(img4d.slice({l}),gKernel3d, xStride, yStride,mapDimens[0],mapDimens[0],false);
    }
    return result;
}

Tensor CnnUtils::normaliseImg(Tensor img,std::vector<float> pixelMeans,std::vector<float> pixelStdDevs){
    std::vector<int> imgDimens = img.getDimens();
    if(imgDimens.size()!=3){
        throw std::invalid_argument("Image must have 3 dimensions for normaliseImg");
    }
    for(int c=0;c<imgDimens[0];c++){
        for(int i=0;i<imgDimens[1];i++){
            for(int j=0;j<imgDimens[2];j++){
                *img[{c,i,j}] = ((*img[{c,i,j}])-pixelMeans[c])/pixelStdDevs[c];
            }
        }
    }
    return img;
}

Tensor CnnUtils::gaussianBlurKernel(int width,int height){ //This will be odd sized
    Tensor kernel({height,width});
    float stdDev = (float)(width+height)/8; //say that items that are half the kernel radius away is the stdDev
    int xCentre = (int)width/2;
    int yCentre = (int)height/2;
    for(int y=0;y<height;y++){
        for(int x=0;x<width;x++){
            (*kernel[{y,x}]) = std::min((float) ((float) (1/(2*M_PI*pow(stdDev,2)))
            *exp(-(pow(x-xCentre,2)+pow(y-yCentre,2))/(2*pow(stdDev,2)))),255.0f);
            //https://en.wikipedia.org/wiki/Gaussian_blur
        }
    }
    return kernel;
}

Tensor CnnUtils::maxPool(Tensor image,int xStride,int yStride){ 
    int xKernelRadius = (int) floor(xStride/2); //Not actually a radius, actually half the width
    int yKernelRadius = (int) floor(yStride/2); 
    std::vector<int> imgDimens = image.getDimens();
    if(imgDimens.size()!=2){
        throw std::invalid_argument("Image must have 2 dimensions for maxPool");
    }
    int imHeight = imgDimens[0];
    int imWidth = imgDimens[1];
    float max;
    int resHeight = (int)floor((float)(imHeight)/yStride);
    int resWidth = (int) floor((float)(imWidth)/xStride);
    Tensor result({resHeight,resWidth});
    int newY,newX = newY =0;
    for(int y=yKernelRadius;y<=imHeight-yKernelRadius;y+=yStride){
        for(int x=xKernelRadius;x<=imWidth-xKernelRadius;x+=xStride){
            max = -std::numeric_limits<float>::infinity();
            for(int j=0;j<yStride;j++){
                for(int i=0;i<xStride;i++){
                    if((*image[{(y+j-yKernelRadius),(x+i-xKernelRadius)}])>max){
                        max = *image[{(y+j-yKernelRadius),(x+i-xKernelRadius)}];
                    }
                }
            }
            *result[{newY,newX}] = max;
            newX++;
        }
        newX=0;
        newY++;
    }
    return result;
}

//variable size output
Tensor CnnUtils::convolution(Tensor image,Tensor kernel,int xStride,int yStride,bool padding){ 
    std::vector<int> imgDimens = image.getDimens();
    std::vector<int> kernelDimens = kernel.getDimens();
    if(imgDimens.size()!=3){
        throw std::invalid_argument("Image must have 3 dimensions for convolution");
    }
    if(kernelDimens.size()!=3){
        throw std::invalid_argument("Kernel must have 3 dimensions for convolution");
    }
    if(kernelDimens[0]!=imgDimens[0]){
        throw std::invalid_argument("The image and kernel must have the same number of channels for convolution");
    }
    int xKernelRadius = (int) floor(kernelDimens[2]/2); //Not actually a radius, actually half the width
    int yKernelRadius = (int) floor(kernelDimens[1]/2);
    float sum;
    Tensor paddedImage(imgDimens);
    if(padding){
        int paddedHeight = imgDimens[1]+yKernelRadius*2;
        int paddedWidth = imgDimens[2]+xKernelRadius*2;
        for(int l=0;l<imgDimens[0];l++){ //for each image channel
            for(int y=yKernelRadius;y<imgDimens[1]+yKernelRadius;y++){
                for(int x=xKernelRadius;x<imgDimens[2]+xKernelRadius;x++){
                    *paddedImage[{l,y,x}] = *image[{l,(y-yKernelRadius),(x-xKernelRadius)}];
                }
            }
        }
    }
    else{
        paddedImage = image; //The assignment operator performs a value by value copy of the data
    }
    std::vector<int> paddedImgDimens = paddedImage.getDimens();
    int imHeight = paddedImgDimens[1]; //assumption that all channels have same dimensions
    int imWidth = paddedImgDimens[2];
    Tensor result({
        (int)ceil((float)(imHeight-2*yKernelRadius)/yStride),
        (int)ceil((float)(imWidth-2*xKernelRadius)/xStride)
    });
    for(int l=0;l<paddedImgDimens[0];l++){
        int newY,newX = newY =0;
        for(int y=yKernelRadius;y<imHeight-yKernelRadius;y+=yStride){
            for(int x=xKernelRadius;x<imWidth-xKernelRadius;x+=xStride){
                sum = 0;
                for(int j=0;j<kernelDimens[1];j++){
                    //TODO figure out biases 
                    for(int i=0;i<kernel[l][0].size();i++){ //[l][0] as the last kernel has the bias which we don't want 
                        sum += kernel[l][j][i] *  paddedImage[l][y+j-yKernelRadius][x+i-xKernelRadius];
                    }
                }
                //Biases
                if(kernel[l][kernel[l].size()-1].size()==kernel[l][0].size()+1){//if we have an extra num on the end it will be the bias
                    sum += kernel[l][kernel[l].size()-1][kernel[l][0].size()]; //this occurs on the last channel
                }
                result[newY][newX] += sum; 
                newX++;
            }
            newX=0;
            newY++;
        }
    }
    for(int y=0;y<result.size();y++){
        for(int x=0;x<result[y].size();x++){
            result[y][x] = leakyRelu(result[y][x]); //has to be here as otherwise we would relu before we've done all the channels
        }
    }
    return result;
}


//fixed size output
Tensor CnnUtils::convolution(Tensor image,Tensor kernel,int xStride,int yStride,int newWidth,int newHeight,bool padding){ 
    //by padding a normal convolution with 0s
    Tensor result = newMatrix({newHeight,newWidth});
    Tensor convResult = convolution(image, kernel, xStride, yStride,padding);
    for(int i=0;i<newHeight;i++){
        for(int j=0;j<newWidth;j++){
            result[i][j] = (i<convResult.size() && j<convResult[i].size())?convResult[i][j]:0;
        }
    }
    return result;
}


//----------------------------------------------------
//MATHS UTILS

std::vector<float> CnnUtils::softmax(std::vector<float> inp){
    std::vector<float> result(inp.size());
    float sum = 0.0f;
    for(int i=0;i<inp.size();i++){
        //e^15 is quite big (roughly 2^22)
        sum += exp(max(min(15,inp[i]),-15)); 
    }
    for(int i=0;i<inp.size();i++){
        result[i] = (float) (exp(max(min(15,inp[i]),-15))/sum);
    }
    return result;
}
        
float CnnUtils::CnnUtils::sigmoid(float num) {
    if (num > 200)
        return 1;
    if (num < -200)
        return 0;
    return 1 / (float) (1 + pow(M_E, -num));
}

float CnnUtils::CnnUtils::relu(float num) {
    if (num <= 0)
        return 0;
    return num;
}

float CnnUtils::CnnUtils::leakyRelu(float num) {
    if (num <= 0)
        return num*0.01f;
    return num;
}

float CnnUtils::CnnUtils::normalDistRandom(float mean,float stdDev){
    std::normal_distribution<double> dist(mean,stdDev);
    std::default_random_engine generator;
    float result = (float) dist(generator);
    return result;
}


//----------------------------------------------------
//UTILS

void CnnUtils::reset(){
    for(int i=0;i<activations.size();i++){
        for(int j=0;j<activations[i].size();j++){
            activations[i][j] = 0;
        }
    }
    for(int l=0;l<numMaps.size();l++){
        for(int i=0;i<numMaps[l];i++){
            for(int j=0;j<mapDimens[l];j++){ 
                for(int k=0;k<mapDimens[l];k++){
                    maps[l][i][j][k] = 0;
                }
            }
        }
    }
}

Tensor CnnUtils::loadKernels(bool loadNew){
    if(loadNew){
        Tensor lKernels = newMatrix({numMaps.size()-1,0,0,0,0});
        for(int l=0;l<lKernels.size();l++){
            if(kernelSizes[l]==0){
                lKernels[l] = newMatrix({0,0,0,0});
            }
            else{
                lKernels[l] = newMatrix({numMaps[l+1],numMaps[l],kernelSizes[l],kernelSizes[l]});
                for(int i=0;i<numMaps[l+1];i++){
                    lKernels[l][i][numMaps[l]-1][kernelSizes[l]-1] = newMatrix({kernelSizes[l]+1}); //Bias 
                    //only 1 bias for each new map
                }
            }
        }
        return lKernels;
    }
    else{
        std::ifstream kernelsFile(currDir+"/plantcnn/src/main/resources/kernels.json");
        nlohmann::json::json jsonKernels;
        kernelsFile >> jsonKernels;
        kernelsFile.close();
        Tensor lKernels = jsonKernels.get<Tensor>();
        return lKernels;
    }
}

Tensor CnnUtils::loadWeights(bool loadNew){
    if(loadNew){
        Tensor lWeights = newMatrix({numNeurons.size()-1,0,0});
        for(int l=0;l<numNeurons.size()-1;l++){
            lWeights[l] = newMatrix({numNeurons[l+1],numNeurons[l]+1});//bias
        }
        return lWeights;
    }
    else{
        std::ifstream weightsFile(currDir+"/plantcnn/src/main/resources/weights.json");
        nlohmann::json::json jsonWeights;
        weightsFile >> jsonWeights;
        weightsFile.close();
        Tensor lWeights = jsonWeights.get<Tensor>();
        return lWeights;
    }
}
       
void CnnUtils::applyGradients(){ //(and reset gradients)
    float adjustedGrad;
    for(int i=0;i<kernels.size();i++){
        for(int j=0;j<kernels[i].size();j++){
            for(int k=0;k<kernels[i][j].size();k++){
                for(int l=0;l<kernels[i][j][k].size();l++){ 
                    for(int m=0;m<kernels[i][j][k][l].size();m++){//10^-10, don't update if basically 0
                        if(!(floatCmp(kernelsGrad[i][j][k][l][m],0f))){
                            if(std::isnan(kernelsGrad[i][j][k][l][m])){
                                std::cout << "NaN kernel gradient i:"+i+" "+" j:"+j+" k:"+k+" l:"+l << std::endl;
                                kernelsGrad[i][j][k][l][m] = 0f;
                                continue;
                            }
                            adjustedGrad = kernelsGrad[i][j][k][l][m] * LR;
                            if(adjustedGrad>10){
                                std::cout << "Very large kernel gradient: "+adjustedGrad << std::endl;
                                adjustedGrad = 0;
                            }
                            kernels[i][j][k][l][m] -= adjustedGrad;
                            kernelsGrad[i][j][k][l][m] = 0f;
                        }    
                    } 
                }
            }
        }
    }
    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[i].size(); j++) {
            for (int k = 0; k < weights[i][j].size(); k++) {
                if(!(floatCmp(weightsGrad[i][j][k],0))){
                    if(std::isnan(weightsGrad[i][j][k])){
                        std::cout << "NaN MLP gradient i:"+i+" "+" j:"+j+" k:"+k << std::endl;
                        weightsGrad[i][j][k] = 0f;
                        continue;
                    }
                    adjustedGrad = weightsGrad[i][j][k] * LR;
                    if(adjustedGrad>10){
                        std::cout << "Very large weight gradient: "+adjustedGrad << std::endl;
                        adjustedGrad = 0;
                    }
                    weights[i][j][k] -= adjustedGrad;
                    weightsGrad[i][j][k] = 0f;
                }
            }
        }
    }
}

void CnnUtils::applyGradients(std::vector<CNN*>& cnns){ //(and reset gradients)
    float adjustedGrad; //this cnn must be included in cnns
    for(int n=0;n<cnns.size();n++){
        for(int i=0;i<kernels.size();i++){
            for(int j=0;j<kernels[i].size();j++){
                for(int k=0;k<kernels[i][j].size();k++){
                    for(int l=0;l<kernels[i][j][k].size();l++){
                        for(int m=0;m<kernels[i][j][k][l].size();m++){
                            if(!(floatCmp(cnns[n]->kernelsGrad[i][j][k][l][m],0f))){
                                if(std::isnan(cnns[n]->kernelsGrad[i][j][k][l][m])){
                                    std::cout << "NaN kernel gradient i:"+i+" "+" j:"+j+" k:"+k+" l:"+l+" n:"+n << std::endl;
                                    cnns[n]->kernelsGrad[i][j][k][l][m] = 0f;
                                    continue;
                                }
                                adjustedGrad = cnns[n]->kernelsGrad[i][j][k][l][m] * LR;
                                if(adjustedGrad>10){
                                    std::cout << "Very large kernel gradient: "+adjustedGrad << std::endl;
                                    adjustedGrad = 0;
                                }
                                kernels[i][j][k][l][m] -= adjustedGrad; //adjust this CNN's weights (as it will be cloned next batch)
                                cnns[n]->kernelsGrad[i][j][k][l][m] = 0f;
                            }   
                        }   
                        
                    }
                }
            }
        }
        for (int i = 0; i < weights.size(); i++) {
            for (int j = 0; j < weights[i].size(); j++) {
                for (int k = 0; k < weights[i][j].size(); k++) {
                    if(!(floatCmp(cnns[n]->weightsGrad[i][j][k],0f))){
                        if(std::isnan(cnns[n]->weightsGrad[i][j][k])){
                            std::cout << "NaN MLP gradient i:"+i+" "+" j:"+j+" k:"+k+" n:"+n << std::endl;
                            cnns[n]->weightsGrad[i][j][k] = 0f;
                            continue;
                        }
                        adjustedGrad = cnns[n]->weightsGrad[i][j][k] * LR;
                        if(adjustedGrad>10){
                            std::cout << "Very large weight gradient: "+adjustedGrad << std::endl;
                            adjustedGrad = 0;
                        }
                        weights[i][j][k] -= adjustedGrad;
                        cnns[n]->weightsGrad[i][j][k] = 0f;
                    }
                }
            }
        }
    }
}

void CnnUtils::resetKernels(){
    for(int i=0;i<kernels.size();i++){ //layer
        for(int j=0;j<kernels[i].size();j++){ //current channel
            for(int k=0;k<kernels[i][j].size();k++){ //previous channel
                int numElems = kernels[i][j].size()*kernels[i][j][k].size()*kernels[i][j][k][0].size(); 
                //num kernels for that layer * h * w
                float stdDev = (float) Math.sqrt((float)2/numElems);
                for(int y=0;y<kernels[i][j][k].size();y++){
                    for(int x=0;x<kernels[i][j][k][y].size();x++){
                        kernels[i][j][k][y][x] = normalDistRandom(0, stdDev); //He initialisation
                    }
                }
            }
            //set the bias = 0
            int finalPrevChannel = kernels[i][j].size()-1;
            int lastY = kernels[i][j][finalPrevChannel].size()-1;
            int lastX = kernels[i][j][finalPrevChannel][lastY].size()-1;
            kernels[i][j][finalPrevChannel][lastY][lastX] = 0; 
        }
    }
    saveKernels();
}

void CnnUtils::resetWeights() {
    for (int i = 0; i < weights.size(); i++) { //layer
        for (int j = 0; j < weights[i].size(); j++) { //neurone
            float stdDev = (float) sqrt((float)2/(weights[i][j].size()-1));
            for (int k = 0; k < weights[i][j].size()-1; k++) { //previous neurone
                weights[i][j][k] = normalDistRandom(0, stdDev);
            }
            weights[i][j][weights[i][j].size()-1] = 0; //bias is set to 0
        }
    }
    saveWeights();
}



void CnnUtils::CnnUtils::saveWeights() {
    std::ofstream weightsFile(currDir+"/plantcnn/src/main/resources/weights.json");
    nlohmann::json::json jsonWeights = weights;
    weightsFile << jsonWeights.dump();
    weightsFile.close();
}

void CnnUtils::saveKernels() {
    std::ofstream kernelsFile(currDir+"/plantcnn/src/main/resources/kernels.json");
    nlohmann::json::json jsonKernels = kernels;
    kernelsFile << jsonKernels.dump();    
    kernelsFile.close();
}

void CnnUtils::saveActivations(){  //For debugging use
    std::ofstream activationsFile(currDir+"/plantcnn/src/main/resources/activations.json");
    nlohmann::json::json jsonActivations = activations;
    activationsFile << jsonActivations.dump();
    activationsFile.close();
}

