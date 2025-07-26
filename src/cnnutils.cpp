#include "cnnutils.hpp"
#include "cnn.hpp" //Needs to be in the .cpp file to avoid a circular dependency but we still need member functions

//----------------------------------------------------
//IMAGE-RELATED
Tensor CnnUtils::parseImg(Tensor& img){
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
        Tensor sliced = img4d.slice({l});
        result.slice({l}) = convolution(sliced,gKernel3d, xStride, yStride,mapDimens[0],mapDimens[0],false);
    }
    return result;
}

Tensor CnnUtils::normaliseImg(Tensor& img,std::vector<float> pixelMeans,std::vector<float> pixelStdDevs){
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
            (*kernel[{y,x}]) = std::min((float) ((float) (1/(2*std::numbers::pi*pow(stdDev,2)))
            *exp(-(pow(x-xCentre,2)+pow(y-yCentre,2))/(2*pow(stdDev,2)))),255.0f);
            //https://en.wikipedia.org/wiki/Gaussian_blur
        }
    }
    return kernel;
}

Tensor CnnUtils::maxPool(Tensor& image,int xStride,int yStride){ 
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
Tensor CnnUtils::convolution(Tensor& image,Tensor& kernel,int xStride,int yStride,bool padding){ 
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
    std::vector<int> paddedImgDimens;
    if(padding){
        int paddedHeight = imgDimens[1]+yKernelRadius*2;
        int paddedWidth = imgDimens[2]+xKernelRadius*2;
        paddedImgDimens = {imgDimens[0],paddedHeight,paddedWidth};
    }
    else{
        paddedImgDimens = imgDimens;
    }
    Tensor paddedImage(paddedImgDimens);
    if(padding){
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
                    for(int i=0;i<kernelDimens[2];i++){ 
                        sum += (*(kernel[{l,j,i}])) *  (*(paddedImage[{l,(y+j-yKernelRadius),(x+i-xKernelRadius)}]));
                    }
                }
                //Biases
                Tensor *biases = kernel.getBiases();
                if(biases->getTotalSize()==1){//for a 3D kernel, there should only 1 bias
                    sum += *((*biases)[0]); 
                }
                else if(biases->getTotalSize()>1){
                    throw std::invalid_argument("Too many biases for a 3D kernel");
                }
                result[newY][newX] += sum; 
                newX++;
            }
            newX=0;
            newY++;
        }
    }
    std::vector<int> resultDimens = result.getDimens();
    for(int y=0;y<resultDimens[0];y++){
        for(int x=0;x<resultDimens[1];x++){
            *result[{y,x}] = leakyRelu(*result[{y,x}]); //has to be here as otherwise we would relu before we've done all the channels
        }
    }
    return result;
}


//fixed size output
Tensor CnnUtils::convolution(Tensor& image,Tensor& kernel,int xStride,int yStride,int newWidth,int newHeight,bool padding){ 
    //by padding a normal convolution with 0s
    Tensor result({newHeight,newWidth});
    Tensor convResult = convolution(image, kernel, xStride, yStride,padding);
    std::vector<int> convResultDimens = convResult.getDimens();
    for(int i=0;i<newHeight;i++){
        for(int j=0;j<newWidth;j++){
            (*result[{i,j}]) = (i<convResultDimens[0] && j<convResultDimens[1])?(*convResult[{i,j}]):0;
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
        sum += exp(std::max(std::min(15.0f,inp[i]),-15.0f)); 
    }
    for(int i=0;i<inp.size();i++){
        result[i] = (float) (exp(std::max(std::min(15.0f,inp[i]),-15.0f))/sum);
    }
    return result;
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
    for(int l=0;l<activations.size();l++){
        size_t activationsLayerSize = activations[l].getTotalSize();
        for(int i=0;i<activationsLayerSize;i++){
            *(activations[l][i]) = 0;
        }
    }
    for(int l=0;l<maps.size();l++){
        size_t mapsLayerSize = maps[l].getTotalSize();
        for(int i=0;i<mapsLayerSize;i++){   
            *(maps[l][i]) = 0;
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
        std::ifstream kernelsFile(currDir+"/res/kernelWeights.json");
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
        std::ifstream kernelBiasesFile(currDir+"/res/kernelBiases.json");
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
        std::ifstream weightsFile(currDir+"/res/mlpWeights.json");
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
        std::ifstream mlpBiasesFile(currDir+"/res/mlpBiases.json");
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
    applyGradient(kernels,kernelsGrad);
    applyGradient(weights,weightsGrad);
}

void CnnUtils::applyGradients(std::vector<CNN*>& cnns){ //(and reset gradients)
    //this cnn must be included in cnns
    for(int n=0;n<cnns.size();n++){
        applyGradient(kernels,(cnns[n]->kernelsGrad));
        applyGradient(weights,(cnns[n]->weightsGrad));
    }
}

void CnnUtils::applyGradient(std::vector<Tensor>& values, std::vector<Tensor>& gradient){ //Main values and biases
    if(values.size()!=gradient.size()){
        throw std::invalid_argument("Values and gradient must have the same number of layers for the gradient to be applied");
    }
    for(int l=0;l<values.size();l++){
        std::vector<int> valuesDimens = values[l].getDimens();
        std::vector<int> gradientDimens = gradient[l].getDimens();
        if(values[l].getTotalSize()!=gradient[l].getTotalSize() || valuesDimens.size()!=gradientDimens.size()){
            throw std::invalid_argument("Tensors must have the dimensions for the gradient to be applied");
        }
        for(int i=0;i<valuesDimens.size();i++){
            if(valuesDimens[i]!=gradientDimens[i]){
                throw std::invalid_argument("Tensors must have the dimensions for the gradient to be applied");
            }
        }
        for(int i=0;i<values[l].getTotalSize();i++){
            float gradVal = *(gradient[l][i]);
            if(!(floatCmp(gradVal,0.0f))){
                if(std::isnan(gradVal)){
                    std::cout << "NaN gradient i: "+std::to_string(i) << std::endl;
                    *(gradient[l][i]) = 0;
                    continue;
                }
                float adjustedGrad = gradVal * LR;
                if(adjustedGrad>10){
                    std::cout << "Very large gradient: "+std::to_string(adjustedGrad) << std::endl;
                    adjustedGrad = 0;
                }
                *(values[l][i]) -= adjustedGrad; //adjust this CNN's weights (as it will be cloned next batch)
                *(gradient[l][i]) = 0;
            }   
        }
    }
    std::vector<Tensor> valuesBiases;
    std::vector<Tensor> gradientBiases;
    for(int i=0;i<values.size();i++){
        Tensor *valLayerBiases = values[i].getBiases();
        Tensor *gradLayerBiases = gradient[i].getBiases();
        if(valLayerBiases!=nullptr){
            //Dereferencing but still has the same shared_ptr and so the original bias values will still be updated
            valuesBiases.push_back(*valLayerBiases);
        }
        if(gradLayerBiases!=nullptr){
            gradientBiases.push_back(*gradLayerBiases);
        }
        if((valLayerBiases==nullptr) != (gradLayerBiases==nullptr)){
            throw std::invalid_argument("Biases must have the same dimensions for the gradient to be applied");
        }
    }
    applyGradient(valuesBiases,gradientBiases);
}

void CnnUtils::resetKernels(){
    for(int l=0;l<kernels.size();l++){ //layer
        std::vector<int> kernelsDimens = kernels[l].getDimens();
        for(int i=0;i<kernelsDimens[0];i++){ //current channel
            //num kernels for that layer * h * w
            int numElems = kernelsDimens[1]*kernelsDimens[2]*kernelsDimens[3]; 
            //He initialisation
            float stdDev = (float) sqrt(2.0f/numElems);
            for(int j=0;j<kernelsDimens[1];j++){ //previous channel
                for(int y=0;y<kernelsDimens[2];y++){
                    for(int x=0;x<kernelsDimens[3];x++){
                        *kernels[l][{i,j,y,x}] = normalDistRandom(0, stdDev); 
                    }
                }
            }
        }
        //set the biases = 0
        Tensor *biases = kernels[l].getBiases();
        size_t biasesSize = biases->getTotalSize();
        for(int i=0;i<biasesSize;i++){
            *(*biases)[i] = 0;
        }
    }
    saveKernels();
}

void CnnUtils::resetWeights() {
    for(int l=0;l<weights.size();l++){ //layer
        std::vector<int> weightsDimens = weights[l].getDimens();
            for (int i=0;i<weightsDimens[0];i++) { //neurone
                //He initialisation
                float stdDev = (float) sqrt(2.0f/weightsDimens[1]);
                for (int k=0;k<weightsDimens[1];k++) { //previous neurone
                    *weights[l][{i,k}] = normalDistRandom(0, stdDev);
                }
            }
        //set the biases = 0
        Tensor *biases = weights[l].getBiases();
        size_t biasesSize = biases->getTotalSize();
        for(int i=0;i<biasesSize;i++){
            *(*biases)[i] = 0;
        }
    }
    saveWeights();
}



void CnnUtils::CnnUtils::saveWeights() {
    d3 weightsVec(weights.size());
    d2 biasesVec(weights.size());
    for(int l=0;l<weights.size();l++){
        weightsVec[l] = weights[l].toVector<d2>();
        biasesVec[l] = weights[l].getBiases()->toVector<d1>();
    }
    
    std::ofstream weightsFile(currDir+"/res/mlpWeights.json");
    nlohmann::json jsonWeights = weightsVec;
    weightsFile << jsonWeights.dump();
    weightsFile.close();

    std::ofstream biasesFile(currDir+"/res/mlpBiases.json");
    nlohmann::json jsonBiases = biasesVec;
    biasesFile << jsonBiases.dump();
    biasesFile.close();
}

void CnnUtils::saveKernels() {
    d5 kernelsVec(kernels.size());
    d2 biasesVec(kernels.size());
    for(int l=0;l<kernels.size();l++){
        kernelsVec[l] = kernels[l].toVector<d4>();
        biasesVec[l] = kernels[l].getBiases()->toVector<d1>();
    };

    std::ofstream kernelsFile(currDir+"/res/kernelWeights.json");
    nlohmann::json jsonKernels = kernelsVec;
    kernelsFile << jsonKernels.dump();    
    kernelsFile.close();

    std::ofstream biasesFile(currDir+"/res/kernelBiases.json");
    nlohmann::json jsonBiases = biasesVec;
    biasesFile << jsonBiases.dump();
    biasesFile.close();
}

void CnnUtils::saveActivations(){  //For debugging use
    std::ofstream activationsFile(currDir+"/res/activations.json");
    d2 activationsVec(activations.size());
    for(int l=0;l<activations.size();l++){
        activationsVec[l] = activations[l].toVector<d1>();
    }
    nlohmann::json jsonActivations = activationsVec;
    activationsFile << jsonActivations.dump();
    activationsFile.close();
}

void CnnUtils::resetGrad(std::vector<Tensor>& grad){
    for(Tensor t:grad){
        size_t size = t.getTotalSize();
        for(size_t i=0;i<size;i++){
            *(t[i]) = 0;
        }
    }
}