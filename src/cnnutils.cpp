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

std::vector<Tensor> CnnUtils::loadKernels(bool loadNew){
    if(loadNew){
        std::vector<Tensor> result(numMaps.size()-1);
        for(int l=0;l<numMaps.size()-1;l++){
            if(kernelSizes[l]==0){
                result[l] = Tensor({0,0,0,0});
            }
            else{
                result[l] = Tensor({numMaps[l+1],numMaps[l],kernelSizes[l],kernelSizes[l]});
                Tensor *biases = new Tensor({numMaps[l+1]});
                //only 1 bias for each new map
                result[l].setBiases(biases);
            }
        }
        return result;
    }
    else{
        std::ifstream kernelsFile(currDir+"/plantcnn/src/main/resources/kernels.json");
        nlohmann::json jsonKernels;
        kernelsFile >> jsonKernels;
        kernelsFile.close();
        d5 kernelsVec = jsonKernels.get<d5>();
        std::vector<Tensor> result(kernelsVec.size());
        //Copy values into tensors
        for(int i=0;i<kernelsVec.size();i++){ //layers
            int numOutChans = kernelsVec[i].size(); //Out channels
            int numInChans = kernelsVec[i][0].size(); //in channels
            int height = kernelsVec[i][0][0].size();
            int width = kernelsVec[i][0][0][0].size();
            result[i] = Tensor({numOutChans,numInChans,height,width});
            for(int j=0;j<numOutChans;j++){
                for(int k=0;k<numOutChans;k++){
                    for(int l=0;l<height;l++){
                        for(int m=0;m<width;m++){
                            *((result[i])[{j,k,l,m}]) = kernelsVec[i][j][k][l][m];
                        }
                    }
                }
            }
        }
        std::ifstream kernelBiasesFile(currDir+"/plantcnn/src/main/resources/kernelBiases.json");
        nlohmann::json jsonBiases;
        kernelBiasesFile >> jsonBiases;
        kernelBiasesFile.close();
        //2d as there's a bias for each output channel in each layer
        d2 biasesVec = jsonBiases.get<d2>();
        if(biasesVec.size()!=kernelsVec.size()){ //i.e. some layers are missing
            throw std::invalid_argument("Number of weights does not match number of biases");
        }
        for(int i=0;i<biasesVec.size();i++){
            Tensor *biases = new Tensor({(int)biasesVec[i].size()});
            for(int j=0;j<biasesVec[i].size();j++){
                *(*biases)[{j}] = biasesVec[i][j];
            }
            result[i].setBiases(biases);
        }
        return result;
    }
}

std::vector<Tensor> CnnUtils::loadWeights(bool loadNew){
    //Each layer of weights is a tensor
    if(loadNew){
        std::vector<Tensor> result((int)numNeurons.size()-1);
        for(int l=0;l<numNeurons.size()-1;l++){
            Tensor layer({numNeurons[l+1],numNeurons[l]});
            Tensor *biases = new Tensor({numNeurons[l+1]});
            layer.setBiases(biases);
            result[l] = layer;
        }
        return result;
    }
    else{
        std::ifstream weightsFile(currDir+"/plantcnn/src/main/resources/weights.json");
        nlohmann::json jsonWeights;
        weightsFile >> jsonWeights;
        weightsFile.close();
        d3 weightsVec = jsonWeights.get<d3>();
        std::vector<Tensor> result(weightsVec.size());
        //Copy values into tensors
        for(int i=0;i<weightsVec.size();i++){
            result[i] = Tensor({(int)weightsVec[i].size(),(int)weightsVec[i][0].size()});
            for(int j=0;j<weightsVec[i].size();j++){
                for(int k=0;k<weightsVec[i][j].size();k++){
                    *(result[i][{j,k}]) = weightsVec[i][j][k];
                }
            }
        }
        std::ifstream mlpBiasesFile(currDir+"/plantcnn/src/main/resources/mlpBiases.json");
        nlohmann::json jsonBiases;
        mlpBiasesFile >> jsonBiases;
        mlpBiasesFile.close();
        d2 biasesVec = jsonBiases.get<d2>();
        if(biasesVec.size()!=weightsVec.size()){ //i.e. some layers are missing
            throw std::invalid_argument("Number of weights does not match number of biases");
        }
        for(int i=0;i<biasesVec.size();i++){
            Tensor *biases = new Tensor({(int)biasesVec[i].size()});
            for(int j=0;j<biasesVec[i].size();j++){
                *(*biases)[{j}] = biasesVec[i][j];
            }
            result[i].setBiases(biases);
        }
        return result;
    }
}
       
void CnnUtils::applyGradients(){ //(and reset gradients)
    for(int i=0;i<kernels.getTotalSize();i++){
        float gradVal = *kernelsGrad[i];
        if(!(floatCmp(gradVal,0.0f))){
            if(std::isnan(gradVal)){
                std::cout << "NaN kernel gradient i: "+std::to_string(i) << std::endl;
                *(kernelsGrad[i]) = 0;
                continue;
            }
            float adjustedGrad = gradVal * LR;
            if(adjustedGrad>10){
                std::cout << "Very large kernel gradient: "+std::to_string(adjustedGrad) << std::endl;
                adjustedGrad = 0;
            }
            *kernels[i] -= adjustedGrad; //adjust this CNN's weights (as it will be cloned next batch)
            *kernelsGrad[i] = 0;
        }   
    }
    for (int i=0;i<weights.getTotalSize();i++) {
        float gradVal = *weightsGrad[i];
        if(!(floatCmp(gradVal,0.0f))){
            if(std::isnan(gradVal)){
                std::cout << "NaN MLP gradient i:"+std::to_string(i) << std::endl;
                *(weightsGrad[i]) = 0;
                continue;
            }
            float adjustedGrad = gradVal * LR;
            if(adjustedGrad>10){
                std::cout << "Very large weight gradient: "+std::to_string(adjustedGrad) << std::endl;
                adjustedGrad = 0;
            }
            *weights[i] -= adjustedGrad;
            *weightsGrad[i] = 0;
        }
    }

    
}

void CnnUtils::applyGradients(std::vector<CNN*>& cnns){ //(and reset gradients)
    //this cnn must be included in cnns
    for(int n=0;n<cnns.size();n++){
        for(int i=0;i<kernels.getTotalSize();i++){
            float gradVal = *(cnns[n]->kernelsGrad[i]);
            if(!(floatCmp(gradVal,0.0f))){
                if(std::isnan(gradVal)){
                    std::cout << "NaN kernel gradient i: "+std::to_string(i) << std::endl;
                    *(cnns[n]->kernelsGrad[i]) = 0;
                    continue;
                }
                float adjustedGrad = gradVal * LR;
                if(adjustedGrad>10){
                    std::cout << "Very large kernel gradient: "+std::to_string(adjustedGrad) << std::endl;
                    adjustedGrad = 0;
                }
                *kernels[i] -= adjustedGrad; //adjust this CNN's weights (as it will be cloned next batch)
                *(cnns[n]->kernelsGrad[i]) = 0;
            }   
        }
        for (int i=0;i<weights.getTotalSize();i++) {
            float gradVal = *(cnns[n]->weightsGrad[i]);
            if(!(floatCmp(gradVal,0.0f))){
                if(std::isnan(gradVal)){
                    std::cout << "NaN MLP gradient i:"+std::to_string(i) << std::endl;
                    *(cnns[n]->weightsGrad[i]) = 0;
                    continue;
                }
                float adjustedGrad = gradVal * LR;
                if(adjustedGrad>10){
                    std::cout << "Very large weight gradient: "+std::to_string(adjustedGrad) << std::endl;
                    adjustedGrad = 0;
                }
                *weights[i] -= adjustedGrad;
                *(cnns[n]->weightsGrad[i]) = 0;
            }
        }
    }
}

void CnnUtils::resetKernels(){
    std::vector<int> kernelsDimens = kernels.getDimens();
    for(int i=0;i<kernelsDimens[0];i++){ //layer
        for(int j=0;j<kernelsDimens[1];j++){ //current channel
            //num kernels for that layer * h * w
            int numElems = kernelsDimens[1]*kernelsDimens[2]*kernelsDimens[3]; 
            //He initialisation
            float stdDev = (float) sqrt((float)2/numElems);
            for(int k=0;k<kernelsDimens[1];k++){ //previous channel
                for(int y=0;y<kernelsDimens[2];y++){
                    for(int x=0;x<kernelsDimens[3];x++){
                        *kernels[{i,j,k,y,x}] = normalDistRandom(0, stdDev); 
                    }
                }
            }
        }
    }
    //set the biases = 0
    Tensor *biases = kernels.getBiases();
    size_t biasesSize = biases->getTotalSize();
    for(int i=0;i<biasesSize;i++){
        *(*biases)[i] = 0;
    }
    saveKernels();
}

void CnnUtils::resetWeights() {
    std::vector<int> weightsDimens = weights.getDimens();
    for (int i=0;i<weightsDimens[0];i++) { //layer
        for (int j=0;j<weightsDimens[1];j++) { //neurone
            //He initialisation
            float stdDev = (float) sqrt(2.0f/weightsDimens[2]);
            for (int k=0;k<weightsDimens[2];k++) { //previous neurone
                *weights[{i,j,k}] = normalDistRandom(0, stdDev);
            }
        }
    }
    //set the biases = 0
    Tensor *biases = weights.getBiases();
    size_t biasesSize = biases->getTotalSize();
    for(int i=0;i<biasesSize;i++){
        *(*biases)[i] = 0;
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

