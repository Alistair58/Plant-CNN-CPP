#include "cnnutils.hpp"
#include "cnn.hpp" //Needs to be in the .cpp file to avoid a circular dependency but we still need member functions

//----------------------------------------------------
//IMAGE-RELATED


//NOTE:
//Inner loops that receive a lot of traffic may look very messy
//This is for performance reasons
//The pretty looking [{i,j,k}] is too slow for these inner loops
//So the raw pointer is used

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
    //If we floor it, we go out of the mapDimens[0] x mapDimens[0] bounds (as we aren't striding far enough)
    int xStride = (int) std::ceil((float)imWidth/mapDimens[0]); //Reducing size to mapDimens[0] x mapDimens[0] via a Gaussian blur
    int yStride = (int) std::ceil((float)imHeight/mapDimens[0]); 
    Tensor gKernel = gaussianBlurKernel(xStride,yStride);
    Tensor gKernel3d = Tensor({1,yStride,xStride});
    gKernel3d.slice({0}) = gKernel;
    Tensor result = Tensor({channels,mapDimens[0],mapDimens[0]});
    Tensor img4d = Tensor({channels,1,imHeight,imWidth}); //convolution requires a 3d array (image with multiple channels) 
    //but we only want to process one channel at a time and so we have to store each channel in a separate 3d array
    for(int l=0;l<channels;l++){
        //Deep copy
        img4d.slice({l,0}) = img.slice({l});
        //Copy-elision
        Tensor sliced = img4d.slice({l});
        //Deep copy
        result.slice({l}) = convolution(sliced,gKernel3d, xStride, yStride,mapDimens[0],mapDimens[0],false);
    }
    return result;
}

void CnnUtils::normaliseImg(Tensor& img,std::vector<float> pixelMeans,std::vector<float> pixelStdDevs){
    std::vector<int> imgDimens = img.getDimens();
    if(imgDimens.size()!=3){
        throw std::invalid_argument("Image must have 3 dimensions for normaliseImg");
    }
    float*  __restrict__ imgData = img.getData();
    std::vector<int> imgChildSizes = img.getChildSizes();
    for(int c=0;c<imgDimens[0];c++){
        int imageChannel = c*imgChildSizes[0];
        for(int i=0;i<imgDimens[1];i++){
            int imageRow = imageChannel + i*imgChildSizes[1];
            for(int j=0;j<imgDimens[2];j++){
                imgData[imageRow+j] = ((imgData[imageRow+j])-pixelMeans[c])/pixelStdDevs[c];
            }
        }
    }
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
    int xKernelRadius = (int) floor(xStride/2); //Not actually a radius, actually half the width of the kernel
    int yKernelRadius = (int) floor(yStride/2); 
    std::vector<int> imgDimens = image.getDimens();
    if(imgDimens.size()!=2){
        throw std::invalid_argument("Image must have 2 dimensions for maxPool");
    }
    int imHeight = imgDimens[0];
    int imWidth = imgDimens[1];
    int resHeight = imHeight/yStride;
    int resWidth = imWidth/xStride;
    Tensor result({resHeight,resWidth});
    int newY,newX = newY =0;

    float*  __restrict__ imageData = image.getData();
    float*  __restrict__ resultData = result.getData();
    for(int y=yKernelRadius;y<=imHeight-yKernelRadius;y+=yStride){
        int resultRow = newY*resWidth;
        for(int x=xKernelRadius;x<=imWidth-xKernelRadius;x+=xStride){
            float max = -std::numeric_limits<float>::infinity();
            for(int j=0;j<yStride;j++){
                int imageRow = (y+j-yKernelRadius)*imWidth +x-xKernelRadius;
                for(int i=0;i<xStride;i++){
                    if((imageData[imageRow+i])>max){ //*image[{(y+j-yKernelRadius),(x+i-xKernelRadius)}]
                        max = imageData[imageRow+i];
                    }
                }
            }
            resultData[resultRow+newX] = max;
            newX++;
        }
        newX=0;
        newY++;
    }
    return result;
}

//variable size output
Tensor CnnUtils::convolution(Tensor& image,Tensor& kernel,const int xStride,const int yStride,bool padding){ 
    #if DEBUG >=2
        uint64_t convStart = getCurrTimeMs();
    #endif 
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
    if(kernelDimens[1]&1==0 || kernelDimens[2]&1==0){
        throw std::invalid_argument("convolution only works for odd-sized kernels");
    }
    const int xKernelRadius = (int) floor(kernelDimens[2]/2); //Not actually a radius, actually half the width
    const int yKernelRadius = (int) floor(kernelDimens[1]/2);
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
    float*  __restrict__ imageData = image.getData();
    float*  __restrict__ paddedImageData = paddedImage.getData();
    std::vector<int> imageChildSizes = image.getChildSizes();
    std::vector<int> paddedImageChildSizes = paddedImage.getChildSizes();
    #if DEBUG >=2
        uint64_t paddingLoopStart = getCurrTimeMs();
    #endif
    if(padding){
        for(int l=0;l<imgDimens[0];l++){ //for each image channel
            int imageChannel = l*imageChildSizes[0];
            int paddedImageChannel = l*paddedImageChildSizes[0]+xKernelRadius; //saving additions
            int yLoopBound = imgDimens[1]+yKernelRadius;
            for(int y=yKernelRadius;y<yLoopBound;y++){
                int imageRow = imageChannel + (y-yKernelRadius)*imageChildSizes[1];
                int paddedImageRow = paddedImageChannel + y*paddedImageChildSizes[1];
                //Memcpy can be vectorised
                //paddedImage: xKernelRadius to paddedWidth-xKernelRadius
                //image: 0 to width
                std::memcpy(
                    paddedImageData + paddedImageRow,
                    imageData + imageRow,
                    sizeof(float) * imgDimens[2]
                );
            }
        }
    }
    else{
        paddedImage = image; //The assignment operator performs a value by value copy of the data
    }
    #if DEBUG >=2
        std::cout << "Padding loop took "+std::to_string(getCurrTimeMs()-paddingLoopStart)+"ms padding was set to "+((padding)?"true":"false") << std::endl;
    #endif
    const int imHeight = paddedImgDimens[1]; //assumption that all channels have same dimensions
    const int imWidth = paddedImgDimens[2];
    Tensor result({
        (int)ceil((float)(imHeight-2*yKernelRadius)/yStride),
        (int)ceil((float)(imWidth-2*xKernelRadius)/xStride)
    }); //0 initialised
    float *kernelData = kernel.getData();
    float*  __restrict__ resultData = result.getData();
    Tensor *biases = kernel.getBiases();
    float bias = 0; //for a 3D kernel, there should only 1 bias
    std::vector<int> kernelChildSizes = kernel.getChildSizes();
    std::vector<int> resultChildSizes = result.getChildSizes();
    if(biases!=nullptr && biases->getTotalSize()==1){
        bias = *((*biases)[0]);
    }
    else if(biases!=nullptr && biases->getTotalSize()>1){
        throw std::invalid_argument("Too many biases for a 3D kernel");
    }
    //No biases is valid
    #if DEBUG >=2
        uint64_t convLoopStart = getCurrTimeMs();
    #endif
    if(kernelDimens[1]==3 &&  kernelDimens[2]==3){
        //unrolled 3x3 version 
        const int originalImgYBound = imHeight-1;
        const int originalImgXBound = imWidth-1;
        const int paddedImgDimens0 = paddedImgDimens[0];
        const int paddedImageChildSizes0 = paddedImageChildSizes[0];
        const int paddedImageChildSizes1 = paddedImageChildSizes[1];
        const int resultChildSizes0 = resultChildSizes[0];
        for(int l=0;l<paddedImgDimens0;l++){
            int newY = 0;
            int newX = 0;
            const int kernelChannel = l*9; //3x3 = 9
            const int paddedImageChannel = l*paddedImageChildSizes0;
            //Need these for scalar tail
            const float k00 = kernelData[kernelChannel + 0];
            const float k01 = kernelData[kernelChannel + 1];
            const float k02 = kernelData[kernelChannel + 2];
            const float k10 = kernelData[kernelChannel + 3];
            const float k11 = kernelData[kernelChannel + 4];
            const float k12 = kernelData[kernelChannel + 5];
            const float k20 = kernelData[kernelChannel + 6];
            const float k21 = kernelData[kernelChannel + 7];
            const float k22 = kernelData[kernelChannel + 8];

            // broadcast kernel scalars to AVX registers once per channel
            const __m256 K00 = _mm256_set1_ps(k00);
            const __m256 K01 = _mm256_set1_ps(k01);
            const __m256 K02 = _mm256_set1_ps(k02);
            const __m256 K10 = _mm256_set1_ps(k10);
            const __m256 K11 = _mm256_set1_ps(k11);
            const __m256 K12 = _mm256_set1_ps(k12);
            const __m256 K20 = _mm256_set1_ps(k20);
            const __m256 K21 = _mm256_set1_ps(k21);
            const __m256 K22 = _mm256_set1_ps(k22);

            for(int y=1;y<originalImgYBound;y+=yStride){
                const int resultRow = newY*resultChildSizes0;
                //y is centre of kernel and so we start at y-1
                const float* __restrict__ paddedRow0Base = paddedImageData+paddedImageChannel+(y-1)*paddedImageChildSizes1;
                const float* __restrict__ paddedRow1Base = paddedRow0Base+paddedImageChildSizes1;
                const float* __restrict__ paddedRow2Base = paddedRow1Base+paddedImageChildSizes1;
                
                int x=1;
                //Process different inputs at once
                //Stop short and we can scalar the rest
                for(;x+xStride*7<=originalImgXBound;x+=xStride*8){
                    //x is the centre of the kernel and so we start at x-1
                    const int xSub1 = x-1; 
                    //indices where the pixels are located
                    const __m256i col0Indices = _mm256_setr_epi32(
                        xSub1,
                        xSub1 + xStride,
                        xSub1 + 2 * xStride,
                        xSub1 + 3 * xStride,
                        xSub1 + 4 * xStride,
                        xSub1 + 5 * xStride,
                        xSub1 + 6 * xStride,
                        xSub1 + 7 * xStride
                    );
                    const __m256i col1Indices = _mm256_add_epi32(col0Indices,_mm256_set1_epi32(1));
                    const __m256i col2Indices = _mm256_add_epi32(col0Indices,_mm256_set1_epi32(2));
                    
                    //Get all of the pixels that will be in the (0,0) position for the convolutions
                    const __m256 R00 = _mm256_i32gather_ps(paddedRow0Base,col0Indices,4);    
                    //And then the (0,1)
                    const __m256 R01 = _mm256_i32gather_ps(paddedRow0Base,col1Indices,4);
                    //etc.
                    const __m256 R02 = _mm256_i32gather_ps(paddedRow0Base,col2Indices,4); 
                    //(1,0)
                    const __m256 R10 = _mm256_i32gather_ps(paddedRow1Base,col0Indices,4);
                    const __m256 R11 = _mm256_i32gather_ps(paddedRow1Base,col1Indices,4);
                    const __m256 R12 = _mm256_i32gather_ps(paddedRow1Base,col2Indices,4);

                    const __m256 R20 = _mm256_i32gather_ps(paddedRow2Base,col0Indices,4);
                    const __m256 R21 = _mm256_i32gather_ps(paddedRow2Base,col1Indices,4);
                    const __m256 R22 = _mm256_i32gather_ps(paddedRow2Base,col2Indices,4);

                    //Compute kernel*image for 8 convolutions at once for each kernel element
                    __m256 acc = _mm256_setzero_ps();
                    acc = _mm256_fmadd_ps(K00,R00,acc);
                    acc = _mm256_fmadd_ps(K01,R01,acc);
                    acc = _mm256_fmadd_ps(K02,R02,acc);
                    acc = _mm256_fmadd_ps(K10,R10,acc);
                    acc = _mm256_fmadd_ps(K11,R11,acc);
                    acc = _mm256_fmadd_ps(K12,R12,acc);
                    acc = _mm256_fmadd_ps(K20,R20,acc);
                    acc = _mm256_fmadd_ps(K21,R21,acc);
                    acc = _mm256_fmadd_ps(K22,R22,acc);
                    //Save the result
                    float* __restrict__ resultPtr = resultData + resultRow + newX;
                    //Load result from previous channels
                    __m256 prev = _mm256_loadu_ps(resultPtr);     
                    //Add our result
                    __m256 sum = _mm256_add_ps(prev, acc);
                    //Save our result
                    _mm256_storeu_ps(resultPtr, sum);
                    newX += 8;
                }
                //scalar tail - remaining outputs for this row
                for (;x<originalImgXBound;x+=xStride) {
                    //x-1 as x is the centre
                    const int row0 = paddedImageChannel + x-1;
                    const int row1 = row0 + paddedImageChildSizes1;
                    const int row2 = row1 + paddedImageChildSizes1;
                    resultData[resultRow + newX] +=
                        k00 * paddedImageData[row0] +     k01 * paddedImageData[row0 + 1] +
                        k02 * paddedImageData[row0 + 2] + k10 * paddedImageData[row1]     +
                        k11 * paddedImageData[row1 + 1] + k12 * paddedImageData[row1 + 2] +
                        k20 * paddedImageData[row2]     + k21 * paddedImageData[row2 + 1] +
                        k22 * paddedImageData[row2 + 2];
                    newX++;
                }
                newX=0;
                newY++;
            }
        }
    }
    else{
        const int originalImgYBound = imHeight-yKernelRadius;
        const int originalImgXBound = imWidth-xKernelRadius;
        const int kernelChildSizes0 = kernelChildSizes[0];
        const int kernelChildSizes1 = kernelChildSizes[1];
        const int paddedImgDimens0 = paddedImgDimens[0];
        const int paddedImageChildSizes0 = paddedImageChildSizes[0];
        const int paddedImageChildSizes1 = paddedImageChildSizes[1];
        const int resultChildSizes0 = resultChildSizes[0];
        const int kernelDimens1 = kernelDimens[1];
        const int kernelDimens2 = kernelDimens[2];
        for(int l=0;l<paddedImgDimens0;l++){
            int newY = 0;
            int newX = 0;
            //Precomputing multiplications
            int kernelChannel = l*kernelChildSizes0;
            int paddedImageChannel = l*paddedImageChildSizes0-xKernelRadius; //saving the subtractions
            for(int y=yKernelRadius;y<originalImgYBound;y+=yStride){
                int resultRow = newY*resultChildSizes0;
                for(int x=xKernelRadius;x<originalImgXBound;x+=xStride){
                    float sum = 0;
                    int paddedImageChannelShortct = paddedImageChannel + x; 
                    for(int j=0;j<kernelDimens1;j++){
                        int kernelRow = kernelChannel + j*kernelChildSizes1;
                        int paddedImageRow = paddedImageChannelShortct + (y+j-yKernelRadius)*paddedImageChildSizes1;
                        float *kernelEndPtr = &kernelData[kernelRow+kernelDimens2];
                        for(float *kernelDataPtr = &kernelData[kernelRow],
                            * __restrict__ paddedImageDataPtr = &paddedImageData[paddedImageRow]
                            ;kernelDataPtr<kernelEndPtr;kernelDataPtr++,paddedImageDataPtr++){
                            sum += (*kernelDataPtr) * (*paddedImageDataPtr);
                        }
                    }
                    resultData[resultRow+newX] += sum; 
                    newX++;
                }
                newX=0;
                newY++;
            }
        }
    }
    

    #if DEBUG >=2
        uint64_t convLoopEnd = getCurrTimeMs();
        std::cout << "Conv loop took "+std::to_string(convLoopEnd-convLoopStart)+"ms" << std::endl;
    #endif 
    std::vector<int> resultDimens = result.getDimens();
    for(int y=0;y<resultDimens[0];y++){
        int resultRow = y*resultChildSizes[0];
        for(int x=0;x<resultDimens[1];x++){
            resultData[resultRow+x] = leakyRelu(resultData[resultRow+x]+bias); //has to be here as otherwise we would relu before we've done all the channels
        }
    }
    #if DEBUG >=2
        std::cout << "Conv function took "+std::to_string(getCurrTimeMs()-convStart)+"ms" << std::endl;
    #endif
    return result;
}


//fixed size output
Tensor CnnUtils::convolution(Tensor& image,Tensor& kernel,int xStride,int yStride,int newWidth,int newHeight,bool padding){ 
    //by padding a normal convolution with 0s
    Tensor convResult = convolution(image, kernel, xStride, yStride,padding);
    std::vector<int> convResultDimens = convResult.getDimens();
    if(convResultDimens[0]==newHeight && convResultDimens[1]==newWidth){
        return convResult;
    }
    Tensor result({newHeight,newWidth}); //The data is 0 initialised
    float*  __restrict__ convResultData = convResult.getData();
    float*  __restrict__ resultData = result.getData();
    //Neither result will have any offsets
    for(int y=0;y<convResultDimens[0];y++){
        int resultRow = y*result.getChildSizes()[0];
        int convResultRow = y*convResult.getChildSizes()[0];
        for(int x=0;x<convResultDimens[1];x++){
            resultData[resultRow+x] = convResultData[convResultRow+x];
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
        


//----------------------------------------------------
//UTILS

void CnnUtils::reset(){
    for(int l=0;l<activations.size();l++){
        size_t activationsLayerSize = activations[l].getTotalSize();
        float *activationLayerData = activations[l].getData();
        memset(
            activationLayerData,
            0,
            sizeof(float)*activationsLayerSize
        );
    }
    for(int l=0;l<maps.size();l++){
        size_t mapsLayerSize = maps[l].getTotalSize();
        float *mapsLayerData = maps[l].getData();
        memset(
            mapsLayerData,
            0,
            sizeof(float)*mapsLayerSize
        );
    }
}

std::vector<Tensor> CnnUtils::loadKernels(bool loadNew){
    #if DEBUG
        uint64_t start = getCurrTimeMs();
    #endif
    if(loadNew){
        std::vector<Tensor> result(numMaps.size()-1);
        for(int l=0;l<(numMaps.size()-1);l++){
            if(kernelSizes[l]==0){
                result[l] = Tensor({0,0,0,0});
            }
            else{
                result[l] = Tensor({numMaps[l+1],numMaps[l],kernelSizes[l],kernelSizes[l]});
                Tensor biases = Tensor({numMaps[l+1]});
                //only 1 bias for each new map
                result[l].setBiases(biases);
            }
        }
        #if DEBUG
            std::cout << "loadKernels (loadNew) took "+std::to_string(getCurrTimeMs()-start)+"ms" << std::endl;
        #endif
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
                for(int k=0;k<numInChans;k++){
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
            throw std::invalid_argument("Number of kernel weights does not match number of kernel biases");
        }
        for(int i=0;i<biasesVec.size();i++){
            Tensor biases = Tensor({(int)biasesVec[i].size()});
            for(int j=0;j<biasesVec[i].size();j++){
                *(biases)[j] = biasesVec[i][j];
            }
            result[i].setBiases(biases);
        }
        #if DEBUG
            std::cout << "loadKernels (loadOld) took "+std::to_string(getCurrTimeMs()-start)+"ms" << std::endl;
        #endif
        return result;
    }
}

std::vector<Tensor> CnnUtils::loadWeights(bool loadNew){
    //Each layer of weights is a tensor
    #if DEBUG
        uint64_t start = getCurrTimeMs();
    #endif
    if(loadNew){
        std::vector<Tensor> result((int)numNeurons.size()-1);
        for(int l=0;l<numNeurons.size()-1;l++){
            Tensor layer({numNeurons[l+1],numNeurons[l]});
            Tensor biases = Tensor({numNeurons[l+1]});
            layer.setBiases(biases);
            result[l] = layer;
        }
        #if DEBUG
            std::cout << "loadWeights (loadNew) took "+std::to_string(getCurrTimeMs()-start)+"ms" << std::endl;
        #endif
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
            throw std::invalid_argument("Number of MLP weights does not match number of MLP biases");
        }
        for(int i=0;i<biasesVec.size();i++){
            Tensor biases = Tensor({(int)biasesVec[i].size()});
            for(int j=0;j<biasesVec[i].size();j++){
                *biases[j] = biasesVec[i][j];
            }
            result[i].setBiases(biases);
        }
        #if DEBUG
            std::cout << "loadWeights (loadOld) took "+std::to_string(getCurrTimeMs()-start)+"ms" << std::endl;
        #endif
        return result;
    }
}
       
void CnnUtils::applyGradients(){ //(and reset gradients)
    #if DEBUG
        uint64_t start = getCurrTimeMs();
    #endif
    applyGradient(kernels,kernelsGrad);
    applyGradient(weights,weightsGrad);
    #if DEBUG
        std::cout << "applyGradients took "+std::to_string(getCurrTimeMs()-start)+"ms" << std::endl;
    #endif
}

void CnnUtils::applyGradients(std::vector<CNN*>& cnns){ //(and reset gradients)
    #if DEBUG
        uint64_t start = getCurrTimeMs();
    #endif
    //this cnn must be included in cnns
    for(int n=0;n<cnns.size();n++){
        applyGradient(kernels,(cnns[n]->kernelsGrad));
        applyGradient(weights,(cnns[n]->weightsGrad));
    }
    #if DEBUG
        std::cout << "applyGradients (multiple CNNs) took "+std::to_string(getCurrTimeMs()-start)+"ms" << std::endl;
    #endif
}

void CnnUtils::applyGradient(std::vector<Tensor>& values, std::vector<Tensor>& gradient){ //Main values and biases
    const float maxGrad = 1;
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
        float*  __restrict__ gradData = gradient[l].getData();
        float*  __restrict__ valuesData = values[l].getData();
        for(int i=0;i<values[l].getTotalSize();i++){
            float gradVal = gradData[i];
            if(!(floatCmp(gradVal,0.0f))){
                if(std::isnan(gradVal)){
                    std::cout << "NaN gradient i: "+std::to_string(i) << std::endl;
                    gradData[i] = 0;
                    continue;
                }
                float adjustedGrad = gradVal * LR;
                if(abs(adjustedGrad)>maxGrad){
                    std::cout << "Very large gradient: "+std::to_string(adjustedGrad) << std::endl;
                    adjustedGrad = (adjustedGrad>0)?maxGrad:-maxGrad;
                }
                valuesData[i] -= adjustedGrad; 
                gradData[i] = 0;
            }   
        }
    }
    std::vector<Tensor> valuesBiases;
    std::vector<Tensor> gradientBiases;
    for(int l=0;l<values.size();l++){
        Tensor *valLayerBiases = values[l].getBiases();
        Tensor *gradLayerBiases = gradient[l].getBiases();
        if((valLayerBiases==nullptr) != (gradLayerBiases==nullptr)){
            throw std::invalid_argument("Biases must have the same dimensions for the gradient to be applied");
        }
        if(valLayerBiases!=nullptr && gradLayerBiases!=nullptr){
            size_t valBiasesSize = valLayerBiases->getTotalSize();
            size_t gradBiasesSize = gradLayerBiases->getTotalSize();
            std::vector<int> valBiasesDimens = valLayerBiases->getDimens();
            std::vector<int> gradBiasesDimens = gradLayerBiases->getDimens();
            if(valBiasesSize!=gradBiasesSize || valBiasesDimens.size()!=gradBiasesDimens.size()){
                throw std::invalid_argument("Biases must have the dimensions for the gradient to be applied");
            }
            for(int i=0;i<valBiasesDimens.size();i++){
                if(valBiasesDimens[i]!=gradBiasesDimens[i]){
                    throw std::invalid_argument("Biases must have the dimensions for the gradient to be applied");
                }
            }
            //I don't think that there ever is an offset but you could provide an input with an offset (i.e. biases is a sub-tensor)
            float* __restrict__ gradBiasesData = gradLayerBiases->getData();
            float* __restrict__ valBiasesData = valLayerBiases->getData();
            for(int i=0;i<valBiasesSize;i++){
                float gradVal = gradBiasesData[i];
                if(!(floatCmp(gradVal,0.0f))){
                    if(std::isnan(gradVal)){
                        std::cout << "NaN bias gradient i: "+std::to_string(i) << std::endl;
                        gradBiasesData[i] = 0;
                        continue;
                    }
                    float adjustedGrad = gradVal * LR;
                    if(abs(adjustedGrad)>maxGrad){
                        std::cout << "Very large bias gradient: "+std::to_string(adjustedGrad) << std::endl;
                        adjustedGrad = (adjustedGrad>0)?maxGrad:-maxGrad;
                    }
                    valBiasesData[i] -= adjustedGrad; 
                    gradBiasesData[i] = 0;
                }   
            }
        }
    }
}

void CnnUtils::resetKernels(){
    #if DEBUG
        uint64_t startTime = getCurrTimeMs();
    #endif
    std::random_device rd{}; //Non-deterministic seeder
    std::mt19937 gen{rd()}; //Mersenne twister 
    for(int l=0;l<kernels.size();l++){ //layer
        float *kernelsData = kernels[l].getData();
        std::vector<int> kernelsDimens = kernels[l].getDimens();
        std::vector<int> kernelsChildDimens = kernels[l].getChildSizes();
        for(int i=0;i<kernelsDimens[0];i++){ //current channel
            //num kernels for that layer * h * w
            int kernelsToChannel = i*kernelsChildDimens[0];
            int numElems = kernelsDimens[1]*kernelsDimens[2]*kernelsDimens[3]; 
            //He initialisation
            float stdDev = (float) sqrt(2.0f/numElems);
            std::normal_distribution<float> dist(0,stdDev);
            for(int j=0;j<kernelsDimens[1];j++){ //previous channel
                int kernelsFromChannel = kernelsToChannel + j*kernelsChildDimens[1];
                for(int y=0;y<kernelsDimens[2];y++){
                    int kernelsRow = kernelsFromChannel + y*kernelsChildDimens[2];
                    for(int x=0;x<kernelsDimens[3];x++){
                        kernelsData[kernelsRow+x] = dist(gen); 
                    }
                }
            }
        }
        //set the biases = 0
        Tensor *biases = kernels[l].getBiases();
        size_t biasesSize = biases->getTotalSize();
        float *biasesData = biases->getData();
        memset(biasesData,0,biasesSize*sizeof(float));
    }
    #if DEBUG
        std::cout << "resetKernels took "+std::to_string(getCurrTimeMs()-startTime)+"ms" << std::endl;
    #endif
    saveKernels();
    
}

void CnnUtils::resetWeights() {
    #if DEBUG
        uint64_t startTime = getCurrTimeMs();
    #endif
    std::random_device rd{}; //Non-deterministic seeder
    std::mt19937 gen{rd()}; //Mersenne twister 
    for(int l=0;l<weights.size();l++){ //layer
        float *weightsData = weights[l].getData();
        std::vector<int> weightsChildSizes = weights[l].getChildSizes();
        std::vector<int> weightsDimens = weights[l].getDimens();
        for (int i=0;i<weightsDimens[0];i++) { //neurone
            int weightsTo = i*weightsChildSizes[0];
            //He initialisation
            float stdDev = (float) sqrt(2.0f/weightsDimens[1]);
            std::normal_distribution<float> dist(0,stdDev);
            for (int j=0;j<weightsDimens[1];j++) { //previous neurone
                weightsData[weightsTo+j] = dist(gen); 
            }
        }
        //set the biases = 0
        Tensor *biases = weights[l].getBiases();
        size_t biasesSize = biases->getTotalSize();
        float *biasesData = biases->getData();
        memset(biasesData,0,biasesSize*sizeof(float));
    }
    #if DEBUG
        std::cout << "resetWeights took "+std::to_string(getCurrTimeMs()-startTime)+"ms" << std::endl;
    #endif
    saveWeights();
    
}



void CnnUtils::CnnUtils::saveWeights() {
    #if DEBUG
        uint64_t start = getCurrTimeMs();
    #endif
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

    #if DEBUG
        std::cout << "saveWeights took "+std::to_string(getCurrTimeMs()-start)+"ms" << std::endl;
    #endif
}

void CnnUtils::saveKernels() {
    #if DEBUG
        uint64_t start = getCurrTimeMs();
    #endif
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

    #if DEBUG
        std::cout << "saveKernels took "+std::to_string(getCurrTimeMs()-start)+"ms" << std::endl;
    #endif
}

void CnnUtils::saveActivations(){  //For debugging use
    d2 activationsVec(activations.size());
    for(int l=0;l<activations.size();l++){
        activationsVec[l] = activations[l].toVector<d1>();
    }
    std::ofstream activationsFile(currDir+"/res/activations.json");
    nlohmann::json jsonActivations = activationsVec;
    activationsFile << jsonActivations.dump();
    activationsFile.close();
}

void CnnUtils::saveMaps(){  //For debugging use
    d4 mapsVec(maps.size());
    for(int l=0;l<maps.size();l++){
        mapsVec[l] = maps[l].toVector<d3>();
    }
    std::ofstream mapsFile(currDir+"/res/maps.json");
    nlohmann::json jsonMaps = mapsVec;
    mapsFile << jsonMaps.dump();
    mapsFile.close();
}

void CnnUtils::resetGrad(std::vector<Tensor>& grad){
    for(Tensor& t:grad){
        size_t size = t.getTotalSize();
        float *tData = t.getData();
        memset(
            tData,
            0,
            sizeof(float)*size
        );
        Tensor *biases = t.getBiases();
        if(biases!=nullptr){
            float *biasesData = biases->getData();
            size_t biasesSize = biases->getTotalSize();
            memset(
                biasesData,
                0,
                sizeof(float)*biasesSize
            );
        }
    }
}