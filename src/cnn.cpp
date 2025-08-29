#include "cnn.hpp"

//----------------------------------------------------
//CONSTRUCTORS 

//Creating a fresh CNN
CNN::CNN(float LR,Dataset *dataset,bool restart){
    numNeurons = {1920,960,47};
    numMaps =     {3,  30,60,120};//includes the result of pooling (except final pooling)
    mapDimens =   {128,64,32,16};
    kernelSizes = {   5, 3, 3  };  //0 represents a pooling layer, the last one is excluded
    strides =     {   2, 2, 2,4}; //pooling strides are included
    padding = true;

    this->d = dataset;
    this->kernels = loadKernels(restart);
    this->weights = loadWeights(restart);
    kernelsGrad = std::vector<Tensor>(kernels.size());
    for(int i=0;i<kernels.size();i++){
        kernelsGrad[i] = Tensor(kernels[i].getDimens()); 
        Tensor biasesGrad(kernels[i].getBiases()->getDimens());
        kernelsGrad[i].setBiases(biasesGrad);
    }
    weightsGrad = std::vector<Tensor>(weights.size());
    for(int i=0;i<weights.size();i++){
        weightsGrad[i] = Tensor(weights[i].getDimens());
        Tensor biasesGrad(weights[i].getBiases()->getDimens());
        weightsGrad[i].setBiases(biasesGrad);
    }
    this->LR = LR;
    this->activations = std::vector<Tensor>(numNeurons.size());
    for(int l=0;l<numNeurons.size();l++){
        activations[l] = Tensor({numNeurons[l]});
    }
    this->maps = std::vector<Tensor>(numMaps.size());
    for(int l=0;l<numMaps.size();l++){
        maps[l] = Tensor({numMaps[l],mapDimens[l],mapDimens[l]});
    }
    for(int l=0;l<kernelSizes.size();l++){
        if(kernelSizes[l]==0){
            int pooledDimen = mapDimens[l]/strides[l];
            maxPoolIndices.push_back(std::unique_ptr<int[]>(new int[numMaps[l]*pooledDimen*pooledDimen]));
        }
    }
    //final pooling
    int finalPooledDimen = mapDimens[mapDimens.size()-1]/strides[strides.size()-1];
    maxPoolIndices.push_back(std::unique_ptr<int[]>(
        new int[numMaps[numMaps.size()-1]*finalPooledDimen*finalPooledDimen]
    ));
    if(restart){
        resetKernels();
        resetWeights();
    }
}

//Creating a copy from a template CNN (I can't call it template)
CNN::CNN(CNN *original,float LR,Dataset *dataset,bool deepCopyWeights) {
    numNeurons = original->numNeurons;
    numMaps = original->numMaps;
    mapDimens = original->mapDimens;
    kernelSizes = original->kernelSizes;
    strides = original->strides;
    padding = original->padding;
    d = dataset; //sharing the same dataset
    if(deepCopyWeights){
        kernels = original->kernels; //copy by value
        weights = original->weights;
    }
    else{ //i.e. shallow copy
        if(original->kernels.size()!=original->kernelsGrad.size()){
            throw std::invalid_argument("kernels and kernelsGrad must have the same number of layers");
        }
        if(original->weights.size()!=original->weightsGrad.size()){
            throw std::invalid_argument("weights and weightsGrad must have the same number of layers");
        }
        this->kernels = std::vector<Tensor>(original->kernels.size());
        for(int i=0;i<original->kernels.size();i++){
            this->kernels[i].shallowCopy(original->kernels[i]);
        }
        this->weights = std::vector<Tensor>(original->weights.size());
        for(int i=0;i<original->weights.size();i++){
            this->weights[i].shallowCopy(original->weights[i]);
        }
    }
    kernelsGrad = std::vector<Tensor>(original->kernelsGrad.size());
    for(int i=0;i<original->kernels.size();i++){
        kernelsGrad[i] = Tensor(original->kernelsGrad[i].getDimens()); 
        Tensor biasesGrad(original->kernelsGrad[i].getBiases()->getDimens());
        kernelsGrad[i].setBiases(biasesGrad);
    }
    weightsGrad = std::vector<Tensor>(original->weightsGrad.size());
    for(int i=0;i<original->weights.size();i++){
        weightsGrad[i] = Tensor(original->weightsGrad[i].getDimens());
        Tensor biasesGrad(original->weightsGrad[i].getBiases()->getDimens());
        weightsGrad[i].setBiases(biasesGrad);
    }
    this->LR = LR;
    this->activations = std::vector<Tensor>(numNeurons.size());
    for(int l=0;l<numNeurons.size();l++){
        activations[l] = Tensor({numNeurons[l]});
    }
    this->maps = std::vector<Tensor>(numMaps.size());
    for(int l=0;l<numMaps.size();l++){
        maps[l] = Tensor({numMaps[l],mapDimens[l],mapDimens[l]});
    }
    for(int l=0;l<kernelSizes.size();l++){
        if(kernelSizes[l]==0){
            int pooledDimen = mapDimens[l]/strides[l];
            maxPoolIndices.push_back(std::unique_ptr<int[]>(new int[numMaps[l]*pooledDimen*pooledDimen]));
        }
    }
    //final pooling
    int finalPooledDimen = mapDimens[mapDimens.size()-1]/strides[strides.size()-1];
    maxPoolIndices.push_back(std::unique_ptr<int[]>(
        new int[numMaps[numMaps.size()-1]*finalPooledDimen*finalPooledDimen]
    ));
}


//----------------------------------------------------
//KEY METHODS 


std::string CNN::forwards(Tensor& imageInt){
    #if DEBUG
        uint64_t startTime = getCurrTimeMs();
    #endif
    reset();
    maps[0] = parseImg(imageInt);
    normaliseImg(maps[0],d->getPixelMeans(),d->getPixelStdDevs());
    #if DEBUG
        uint64_t startConvLayers = getCurrTimeMs();
    #endif
    //Convolutional and pooling layers
    for(int l=1;l<numMaps.size();l++){
        for(int i=0;i<numMaps[l];i++){
            //Does copy-elision and so no ctor is called and memory is shared
            Tensor currChannel = maps[l].slice({i}); 
            if(kernelSizes[l-1]==0){
                //1:1 mapping for a max pool layer
                Tensor prevChannel = maps[l-1].slice({i});
                currChannel = maxPool(prevChannel,strides[l-1],strides[l-1]); //maxPool requires 1:1 channels between layers
            }
            else{   
                //Slice with biases
                Tensor kernel = kernels[l-1].slice({i},{i});
                #if DEBUG >= 2
                    uint64_t convStart = getCurrTimeMs();
                #endif
                currChannel = convolution(maps[l-1],kernel,strides[l-1],strides[l-1],padding);
                #if DEBUG >= 2 
                    std::cout << "Convolutional layer "+std::to_string(l)+" channel "+std::to_string(i)+" took "+std::to_string(getCurrTimeMs()-convStart)+"ms" << std::endl;
                #endif 
            }
        }
    }
    #if DEBUG
        uint64_t endConvLayers = getCurrTimeMs();
        std::cout << "Convolutional layers took "+std::to_string(endConvLayers-startConvLayers)+"ms" << std::endl;
    #endif
    //Final pooling 
    int poolingDimen = mapDimens[mapDimens.size()-1]/strides[strides.size()-1];
    int poolingArea = poolingDimen*poolingDimen;
    Tensor pooled({numMaps[numMaps.size()-1],poolingDimen,poolingDimen});
    float *pooledData = pooled.getData();
    float *activations0Data = activations[0].getData();
    std::vector<int> pooledChildSizes = pooled.getChildSizes();
    for(int i=0;i<numMaps[numMaps.size()-1];i++){
        Tensor pooledChannel = pooled.slice({i});
        Tensor prevChannel = maps[numMaps.size()-1].slice({i});
        int *maxPoolIndicesMap = &(maxPoolIndices[maxPoolIndices.size()-1][i*poolingArea]);
        pooledChannel = maxPool(prevChannel,strides[strides.size()-1],strides[strides.size()-1],maxPoolIndicesMap);
        int activationsPoolingArea = i*poolingArea;
        int poolingChannel = i*pooledChildSizes[0];
        for(int y=0;y<poolingDimen;y++){
            int activationsPoolingRow = activationsPoolingArea + y*poolingDimen;
            int poolingRow = poolingChannel + y*pooledChildSizes[1];
            std::memcpy(
                activations0Data+activationsPoolingRow,
                pooledData+poolingRow,
                poolingDimen*sizeof(float)
            );
        }
    }
    //MLP
    for(int l=0;l<weights.size();l++){
        float *biasesData = weights[l].getBiases()->getData();
        float *prevActivations = activations[l].getData();
        float *currActivations = activations[l+1].getData();
        float *currWeights = weights[l].getData();
        for(int i=0;i<numNeurons[l+1];i++){
            int weightsTo = i*numNeurons[l];
            for(int j=0;j<numNeurons[l];j++){
                //likely quicker with axv2
                currActivations[i] += prevActivations[j] * currWeights[weightsTo+j]; 
            }
            currActivations[i] += biasesData[i]; //add bias
            if(l!=weights.size()-1){ //We'll softmax the last layer and so relu is unnecessary
                currActivations[i]= leakyRelu(currActivations[i]);
            }
        }
    }
    activations[activations.size()-1] = softmax(activations[activations.size()-1].toVector<d1>());
    float largestActivation = *(activations[activations.size()-1][0]);
    int result = 0;
    for(int i=1;i<numNeurons[numNeurons.size()-1];i++){
        if(*(activations[activations.size()-1][i])>largestActivation){
            largestActivation = *(activations[activations.size()-1][i]);
            result = i;
        }
    }
    #if DEBUG
        std::cout << "MLP took "+std::to_string(getCurrTimeMs()-endConvLayers)+"ms" << std::endl;

        d1 outputVec = activations[activations.size()-1].toVector<d1>();
        std::cout << "[";
        for(int i=0;i<outputVec.size()-1;i++){
            std::cout << std::to_string(outputVec[i])+",";
        }
        std::cout << std::to_string(outputVec[outputVec.size()-1])+"]" << std::endl;
        std::cout << "Forwards took "+std::to_string(getCurrTimeMs()-startTime)+"ms" <<std::endl;
    #endif
    #if DEBUG >= 2
        saveMaps();
        saveActivations();
    #endif
    return d->plantNames[result];
} 

void CNN::backwards(Tensor& imageInt,std::string answer){ //adds the gradient to its internal gradient arrays
    forwards(imageInt); //set all the activations
    //Gradients are not reset each time to enable batches
    #if DEBUG
        uint64_t mlpStart = getCurrTimeMs();
    #endif
    //MLP derivs
    if(!(d->plantToIndex.contains(answer))){
        std::cout << "\""+answer+"\" does not exist" << std::endl;
        return; 
    }
    int correctOutput = d->plantToIndex[answer];
    
    std::vector<Tensor> dcDzs(numNeurons.size()); //z is the pre-activation summations 
    //The derivative includes the activation derivative
    //z_i = w_j_i*a_j + ... + b_i
    for(int l=0;l<numNeurons.size();l++){
        dcDzs[l] = Tensor({numNeurons[l]}); //all layers need activation derivatives
    }
    int lastLayer = numNeurons.size()-1;
    for(int i=0;i<numNeurons[lastLayer];i++){
        if(std::isnan(*activations[lastLayer][i])){
            std::cout << "Invalid activation in last layer at i:"+std::to_string(i) << std::endl;
            return;
        }
        //Cross entropy loss
        *dcDzs[lastLayer][i] = *activations[lastLayer][i] - ((i==correctOutput)?1:0); 
    }

    mlpBackwards(dcDzs); 
    #if DEBUG
        std::cout << "Backwards MLP took "+std::to_string(getCurrTimeMs()-mlpStart)+"ms" << std::endl;
    #endif
    //x is the image pixel value and so these dcDxs are the derivatives based on pixels which are carried backwards
    std::vector<Tensor> dcDxs(numMaps.size()-2);
    //No dcDxs for first or last (last goes straight into the MLP)
    for(int l=0;l<numMaps.size()-2;l++){
        if(kernelSizes[l+1]==0){
            //There doesn't need to be any dcDxs for any pre-pooling maps
            //Blank dcDxs for pre-pooling layers so that indices stay consistent with map indices
            dcDxs[l] = Tensor({0});
        }
        else{
            dcDxs[l] = Tensor({numMaps[l+1],mapDimens[l+1],mapDimens[l+1]});
        }
    }
    #if DEBUG
       uint64_t convolutionStart = getCurrTimeMs();
    #endif
    //makes computational sense to do pooling and conv together
    finalPoolingConvBackwards(dcDzs,dcDxs,padding);
    #if DEBUG 
        uint64_t prevConvEnd = getCurrTimeMs();
        std::cout << "finalPoolingConvBackwards took "+std::to_string(prevConvEnd-convolutionStart)+"ms" << std::endl;
    #endif
    for(int l=numMaps.size()-2;l>0;l--){ //>0 is due to the input dimens being included in numMaps and -2 as we've already done the last layer
        if(kernelSizes[l-1]==0){
            poolingConvBackwards(dcDxs, --l,padding); //prev (l-1) --conv-> curr (l) --pool-> pooled (l+1)
            //skip 1 layer as we have done it within poolingConvBackwards
            #if DEBUG
                std::cout << "poolingConvBackwards took "+std::to_string(getCurrTimeMs()-prevConvEnd)+"ms" << std::endl;
                prevConvEnd = getCurrTimeMs(); 
            #endif
        }
        else{
            convBackwards(dcDxs,l,padding); //prev (l-1) --conv-> curr (l)
            #if DEBUG 
                std::cout << "convBackwards took "+std::to_string(getCurrTimeMs()-prevConvEnd)+"ms" << std::endl;
                prevConvEnd = getCurrTimeMs(); 
            #endif
        }
        
    }
    #if DEBUG
        std::cout << "Backwards Convolution took "+std::to_string(getCurrTimeMs()-convolutionStart)+"ms" << std::endl;
        std::cout << "Backwards took "+std::to_string(getCurrTimeMs()-mlpStart)+"ms" << std::endl;
    #endif
}


//----------------------------------------------------
//BACKPROPAGATION-RELATED

void CNN::mlpBackwards(std::vector<Tensor>& dcDzs){
    for(int l=weights.size()-1;l>=0;l--){
        float *weightsGradData = weightsGrad[l].getData();
        float *biasesGradData = weightsGrad[l].getBiases()->getData();
        float*  __restrict__ nextDcDzsData = dcDzs[l+1].getData();
        float*  __restrict__ currDcDzsData = dcDzs[l].getData();
        float*  __restrict__ activationsData = activations[l].getData();
        float *weightsData = weights[l].getData();
        for(int i=0;i<numNeurons[l+1];i++){
            int weightsNeuron = i*numNeurons[l];
            for(int j=0;j<numNeurons[l];j++){//NOTE: Weights gradient != negative gradient
                //Can be AX2'd
                weightsGradData[weightsNeuron+j] += (nextDcDzsData[i]) * (activationsData[j]);
                //dC/dw = dC/da_i+1 * da_i+1/dz * dz/dw
                currDcDzsData[j] +=  (nextDcDzsData[i]) * (weightsData[weightsNeuron+j]) * (((activationsData[j])<=0)?0.01f:1);//next layer
                //dC/dz_i = dC/dz_i+1 * dz_i+1/da_i * da_i/dz_i
            }
            //bias
            biasesGradData[i] += nextDcDzsData[i];
        }
    }
}

void CNN::convBackwards(std::vector<Tensor>& dcDxs, int l,bool padding){
    //We are working from the back -> front 
    //Prev is the thing closer to the input image and curr is closer the output vector
    //z = sum(k*x)+b
    //x = ReLU(z)
    const int lSub1 = l-1;
    const int lSub2 = l-2;
    const int prevDimens = mapDimens[lSub1];
    const int currDimens = mapDimens[l];
    const int kernelSize = kernelSizes[lSub1];
    const int kernelRadius = (int) floor(kernelSize/2);
    const int thisStride = strides[lSub1];
    float*  __restrict__ currDcDxsData = dcDxs[lSub1].getData(); //yes, l-1 is correct (dcDxs only has numMaps.size()-2 layers)
    float*  __restrict__ prevDcDxsData = nullptr;
    if(l!=1) prevDcDxsData = dcDxs[lSub2].getData(); //No derivatives need to be stored for the first layer
    float*  __restrict__ currMapData = maps[l].getData();
    float*  __restrict__ prevMapData = maps[lSub1].getData();
    float *kernelData = kernels[lSub1].getData();
    float *kernelGradData = kernelsGrad[lSub1].getData(); 
    float *kernelBiasesGradData = kernelsGrad[lSub1].getBiases()->getData(); //only 1 for each channel (1d)
    std::vector<int> currMapsChildSizes = maps[l].getChildSizes();
    std::vector<int> prevMapsChildSizes = maps[lSub1].getChildSizes();
    std::vector<int> kernelsChildSizes = kernels[lSub1].getChildSizes();
    float buffer[8];
    //Precompute ReLU derivatives so we don't recompute them multiple times
    const size_t currMapSize = maps[l].getTotalSize();
    float* __restrict__ currDcDzsData = (float*) malloc(currMapSize*sizeof(float));
    if(!currDcDzsData){
        throw std::runtime_error("Failed malloc in convBackwards");
    }
    const float * __restrict__ currMapDataEndPtr = currMapData+currMapSize;
    for(
        float* __restrict__ currDcDzsPtr = currDcDzsData,
        * __restrict__ currMapDataPtr = currMapData,
        * __restrict__ currDcDxsPtr = currDcDxsData;
        currMapDataPtr<currMapDataEndPtr;
        currDcDzsPtr++,currMapDataPtr++,currDcDxsPtr++ //All the same dimensions
    ){
        //Can be AVX2'd if necessary
        *currDcDzsPtr = (((*currMapDataPtr)<=0) ? 0.01f : 1.0f) * (*currDcDxsPtr);
    }

    for(int i=0;i<numMaps[l];i++){ //For each convolution output
        int currMapChannel = i*currMapsChildSizes[0];
        int kernelToChannel = i*kernelsChildSizes[0]; //kernels are [layer][nextLayerChannel][prevLayerChannel]
        for(int prevMapI=0;prevMapI<numMaps[lSub1];prevMapI++){ //For each previous channel
            int prevMapChannel = prevMapI*prevMapsChildSizes[0];
            int kernelFromChannel = kernelToChannel + prevMapI*kernelsChildSizes[1];
            for(int j=0;j<kernelSize;j++){
                int kernelRow = kernelFromChannel + j*kernelsChildSizes[2];
                int yStart, yEnd;
                if(padding){
                    yStart = (j<kernelRadius)? floorMod((j-kernelRadius),thisStride) : j-kernelRadius; //want modulus (positive) not the remainder
                    yEnd = std::min(prevDimens-kernelRadius+j,prevDimens); //When j>=kernelRadius, it reaches the end item. We don't care about the stride as this is the upper bound
                }
                else{
                    yStart = j;
                    yEnd = prevDimens-kernelSize+j+1;
                }
                for(int k=0;k<kernelSize;k++){ //For each element in the kernel (k,j)
                    //Add up all the activations that it sees
                    int kernelIndex = kernelRow + k;
                    float kernelVal = kernelData[kernelIndex];
                    float sum = 0;
                    int thisY,thisX;
                    thisY = thisX = 0;
                    
                    int xStart, xEnd;
                    if(padding){
                        xStart = (k<kernelRadius)? floorMod((k-kernelRadius),thisStride) : k-kernelRadius;
                        xEnd = std::min(prevDimens-kernelRadius+k,prevDimens); //Same here - makes sense with a drawing
                        //The limits are needed as we have removed the padding and so we have to stop it earlier
                    }
                    else{
                        xStart = k;
                        xEnd = prevDimens-kernelSize+k+1;
                    }
                    for(int y=yStart;y<yEnd;y+=thisStride){  //For every pixel in the previous layer (x,y) which then corresponds to one in the current (x-k,y-j)
                        int currMapRow = currMapChannel + thisY*currMapsChildSizes[1];
                        int prevMapRow = prevMapChannel + y*prevMapsChildSizes[1];
                        float* __restrict__ currMapDcDzsRowBase = currDcDzsData+currMapRow;
                        int x=xStart;
                        for(;x+7*thisStride<xEnd;x+=8*thisStride){
                            int basePrevMapIndex = prevMapRow+x;
                            const __m256i prevMapIndices = _mm256_setr_epi32(
                                basePrevMapIndex,
                                basePrevMapIndex+thisStride,
                                basePrevMapIndex+thisStride*2,
                                basePrevMapIndex+thisStride*3,
                                basePrevMapIndex+thisStride*4,
                                basePrevMapIndex+thisStride*5,
                                basePrevMapIndex+thisStride*6,
                                basePrevMapIndex+thisStride*7
                            );
                            float *currMapDcDzsBasePtr = currMapDcDzsRowBase + thisX;
                            const __m256 prevMapVals = _mm256_i32gather_ps(prevMapData,prevMapIndices,4);      
                            const __m256 currMapDerivs = _mm256_loadu_ps(currMapDcDzsBasePtr);
                            
                            //Add it (dC/dx*dx/dk) to kernel derivative
                            sum += dotProduct8f(prevMapVals,currMapDerivs);
                            
                            if(l!=1){
                                __m256 kernelVals = _mm256_set1_ps(kernelVal);
                                __m256 product = _mm256_mul_ps(currMapDerivs,kernelVals);
                                _mm256_storeu_ps(buffer,product);
                                //Can't store non-contiguously in avx256
                                float* __restrict__ prevDcDxsPtr = prevDcDxsData+basePrevMapIndex;
                                for(int t=0;t<8;t++){ 
                                    *(prevDcDxsPtr+t*thisStride) += buffer[t];
                                }
                            }
                            thisX+=8;
                        }
                        //scalar tail
                        for(;x<xEnd;x+=thisStride){
                            int currMapIndex = currMapRow + thisX;
                            int prevMapIndex = prevMapRow + x;
                                const float reusable = currDcDzsData[currMapIndex];
                                sum += (prevMapData[prevMapIndex]) * reusable;//The previous activation
                                if(l!=1) prevDcDxsData[prevMapIndex] += reusable * kernelVal; //don't have dcDxs for the first layer
                            thisX++;
                        }
                        thisX = 0;
                        thisY++;
                    }
                    kernelGradData[kernelIndex] += sum; 
                }
            }
        }
        //Bias doesn't care about the inputs and so only needs the output
        //Bias has to be here as otherwise it would count the same pixels multiple times
        //Bias deriv = cost deriv * relu deriv * 1 (only 1 bias term in each new pixel expression)
        float biasSum = 0;
        float* __restrict__ currDcDzsPtr = currDcDzsData+currMapChannel;
        const float* __restrict__ currDcDzsEndPtr = currDcDzsPtr+currMapsChildSizes[0];
        for(;currDcDzsPtr<currDcDzsEndPtr;currDcDzsPtr++){
            biasSum += *currDcDzsPtr;
        }
        kernelBiasesGradData[i] += biasSum;
    }
    free(currDcDzsData);
}

void CNN::finalPoolingConvBackwards(std::vector<Tensor>& dcDzs,std::vector<Tensor>& dcDxs,bool padding){
    int lastMapsL = maps.size()-1;
    int prevMapsL = maps.size()-2;
    int lastKernelsL = kernels.size()-1;
    Tensor *kernelBiasesGrad = kernelsGrad[lastKernelsL].getBiases(); //only 1 for each channel (1d)
    float*  __restrict__ activations0Data = activations[0].getData();
    float*  __restrict__ lastMapData = maps[lastMapsL].getData();
    float*  __restrict__ prevMapsData = maps[prevMapsL].getData();
    float*  __restrict__ dcDzs0Data = dcDzs[0].getData();
    float*  __restrict__ lastDcDxsData = nullptr;
    const int dcDxsSize = dcDxs.size();
    //Scenario where there's only 1 conv and then final pooling (doesn't occur in my model - only debugging ones)
    if(dcDxsSize>0) lastDcDxsData = dcDxs[dcDxs.size()-1].getData();
    float *kernelData = kernels[lastKernelsL].getData();
    float *kernelGradData = kernelsGrad[lastKernelsL].getData();
    float *kernelBiasesGradData = kernelsGrad[lastKernelsL].getBiases()->getData(); //only 1 for each channel (1d)
    int* __restrict__ maxPoolIndicesData = maxPoolIndices[maxPoolIndices.size()-1].get();
    int prevDimens = mapDimens[prevMapsL];
    int currDimens = mapDimens[lastMapsL];
    int kernelSize = kernelSizes[lastKernelsL];
    int kernelRadius = (int) floor(kernelSize/2);
    int poolStride = strides[strides.size()-1];
    int thisStride = strides[strides.size()-2];
    int poolWidth = mapDimens[lastMapsL]/strides[strides.size()-1];
    int poolArea = poolWidth*poolWidth;
    std::vector<int> lastMapsChildSizes = maps[lastMapsL].getChildSizes();
    std::vector<int> prevMapsChildSizes = maps[prevMapsL].getChildSizes();
    std::vector<int> lastKernelsChildSizes = kernels[lastKernelsL].getChildSizes();
    //don't count the max pixel more than once
    //ChatGPT says uint8_t is quicker than bool as bool does bit packing
    for(int i=0;i<numMaps[lastMapsL];i++){ //for each final map
        const int mlpRegion = i*poolArea; 
        int lastMapChannel = i*lastMapsChildSizes[0];
        int kernelToChannel = i*lastKernelsChildSizes[0];
        for(int prevMapI=0;prevMapI<numMaps[prevMapsL];prevMapI++){
            int prevMapChannel = prevMapI*prevMapsChildSizes[0];
            int kernelFromChannel = kernelToChannel + prevMapI*lastKernelsChildSizes[1];
            for(int j=0;j<kernelSize;j++){
                int kernelRow = kernelFromChannel + j*lastKernelsChildSizes[2];
                int yStart, yEnd;
                if(padding){
                    yStart = (j<kernelRadius)? floorMod((j-kernelRadius),thisStride) : j-kernelRadius; //want modulus (positive) not the remainder
                    yEnd = std::min(prevDimens-kernelRadius+j,prevDimens); //When j>=kernelRadius, it reaches the end item. We don't care about the stride as this is the upper bound
                }
                else{
                    yStart = j;
                    yEnd = prevDimens-kernelSize+j+1;
                }

                for(int k=0;k<kernelSize;k++){ //For each element in the kernel (k,j)
                    int kernelIndex = kernelRow + k;
                    //Add up all the activations that it sees
                    float sum = 0;
                    int xStart, xEnd;
                    if(padding){
                        xStart = (k<kernelRadius)? floorMod((k-kernelRadius),thisStride) : k-kernelRadius;
                        xEnd = std::min(prevDimens-kernelRadius+k,prevDimens); //Same here - makes sense with a drawing
                        //The limits are needed as we have removed the padding and so we have to stop it earlier
                    }
                    else{
                        xStart = k;
                        xEnd = prevDimens - kernelSize+k+1;
                    }
                    for(int r=0;r<poolArea;r++){
                        int mlpIndex = mlpRegion + r;
                        int maxPixelIndex = maxPoolIndicesData[mlpIndex];
                        //Curr layer indices
                        int thisY = maxPixelIndex/currDimens;
                        int thisX = maxPixelIndex - thisY*currDimens;
                        //Prev layer indices
                        int y = yStart + thisY*thisStride;
                        int x = xStart + thisX*thisStride;
                        if (y >= yEnd || x >= xEnd) {
                            //If this kernel element doesn't touch a real pixel
                            //Occurs when we've padded and so x and y are out of bounds (in the padding)
                            //We set xStart and yStart such that it can't happen at the start
                            continue;
                        }      
                        int lastMapIndex = lastMapChannel + thisY*lastMapsChildSizes[1] + thisX;
                        int prevMapIndex = prevMapChannel + y*prevMapsChildSizes[1] + x;
                        //In the first MLP layer a=relu(x) where x is the max activation pixel from pooling
                        sum+= prevMapsData[prevMapIndex] * dcDzs0Data[mlpIndex]; //The activation of the previous layer * the correct derivative from pooling
                        //Conditional as otherwise we would go out of bounds
                        if(dcDxsSize>0) lastDcDxsData[prevMapIndex] += dcDzs0Data[mlpIndex] * kernelData[kernelIndex];//*kernel weight
                    }
                    kernelGradData[kernelIndex] += sum;
                }
            }
        }
        //Bias has to be here as otherwise it would count the same pixels multiple times
        float biasSum = 0.0f;
        for (int r=0;r<poolArea;r++) {
            const int mlpIndex = mlpRegion + r;
            biasSum += dcDzs0Data[mlpIndex];//Bias deriv = cost deriv * relu deriv * 1 (only 1 bias term in each new pixel expression)
        }
        kernelBiasesGradData[i] += biasSum;
    }
}

void CNN::poolingConvBackwards(std::vector<Tensor>& dcDxs, int l,bool padding){
    int lSub1 = l-1;
    int lPlus1 = l+1;
    int prevDimens = mapDimens[lSub1];
    int currDimens = mapDimens[l];
    int kernelSize = kernelSizes[lSub1];
    int kernelRadius = (int) floor(kernelSize/2);
    int poolStride = strides[l];
    int thisStride = strides[lSub1];
    int poolDimens = mapDimens[lPlus1];
    float*  __restrict__ currMapData = maps[l].getData(); 
    float*  __restrict__ prevMapData = maps[lSub1].getData();
    float*  __restrict__ pooledMapData = maps[lPlus1].getData();
    float*  __restrict__ pooledDcDxsData = dcDxs[l].getData();
    float*  __restrict__ prevDcDxsData = dcDxs[l-2].getData();
    float *kernelData = kernels[lSub1].getData();
    float *kernelGradData = kernelsGrad[lSub1].getData();
    float *kernelBiasesGradData = kernelsGrad[lSub1].getBiases()->getData();
    std::vector<int> currMapsChildSizes = maps[l].getChildSizes();
    std::vector<int> prevMapsChildSizes = maps[lSub1].getChildSizes();
    std::vector<int> pooledMapsChildSizes = maps[lPlus1].getChildSizes();
    std::vector<int> kernelsChildSizes = kernels[lSub1].getChildSizes();
    for(int i=0;i<numMaps[l];i++){ //prev (l-1) --conv-> curr (l) --pool-> pooled (l+1)
        //pooling is 1:1 between channels
        int currMapChannel = i*currMapsChildSizes[0];
        int pooledMapChannel = i*pooledMapsChildSizes[0];
        int kernelToChannel = i*kernelsChildSizes[0];
        for(int prevMapI=0;prevMapI<numMaps[l-1];prevMapI++){
            int kernelFromChannel = kernelToChannel + prevMapI*kernelsChildSizes[1];
            int prevMapChannel = prevMapI*prevMapsChildSizes[0];
            for(int j=0;j<kernelSize;j++){
                int kernelRow = kernelFromChannel + j*kernelsChildSizes[2];
                int yStart, yEnd;
                if(padding){
                    yStart = (j<kernelRadius)? floorMod((j-kernelRadius),thisStride) : j-kernelRadius; //want modulus (positive) not the remainder
                    yEnd = std::min(prevDimens-kernelRadius+j,prevDimens); //When j>=kernelRadius, it reaches the end item. We don't care about the stride as this is the upper bound
                }
                else{
                    yStart = j;
                    yEnd = prevDimens-kernelSize+j+1;
                }
                for(int k=0;k<kernelSize;k++){ //For each element in the kernel (k,j)
                    int kernelIndex = kernelRow + k;
                    //Add up all the activations that it sees
                    float sum = 0;
                    int thisY,thisX;
                    thisY = thisX = 0;
                    std::vector<std::vector<bool>> done(poolDimens,std::vector<bool>(poolDimens));
                    int xStart, xEnd;
                    if(padding){
                        xStart = (k<kernelRadius)? floorMod((k-kernelRadius),thisStride) : k-kernelRadius;
                        xEnd = std::min(prevDimens-kernelRadius+k,prevDimens); //Same here - makes sense with a drawing
                        //The limits are needed as we have removed the padding and so we have to stop it earlier
                    }
                    else{
                        xStart = k;
                        xEnd = prevDimens - kernelSize+k+1;
                    }
                    for(int y=yStart;y<yEnd;y+=thisStride){  //For every pixel in the previous layer (x,y) which then corresponds to one in the current (x-k,y-j)
                        int poolY = ((thisY)/poolStride);
                        int currMapRow = currMapChannel + thisY*currMapsChildSizes[1];
                        int pooledMapRow = pooledMapChannel + poolY*pooledMapsChildSizes[1];
                        int prevMapRow = prevMapChannel + y*prevMapsChildSizes[1];
                        for(int x=xStart;x<xEnd;x+=thisStride){ //Derivatve of the corresponding pixel in the next (backwards) layer
                            int poolX = ((thisX)/poolStride);
                            int pooledMapIndex = pooledMapRow+poolX;
                            int currMapIndex = currMapRow+thisX;
                            int prevMapIndex = prevMapRow+x;
                            if(floatCmp(currMapData[currMapIndex],pooledMapData[pooledMapIndex]) && !done[poolY][poolX] && !floatCmp(pooledDcDxsData[pooledMapIndex],0.0f)){ //only the max element has a derivative
                                done[poolY][poolX] = true;
                                float reusable =  pooledDcDxsData[pooledMapIndex]//Previous derivative (from pooling)
                                * ((currMapData[currMapIndex])<=0?0.01f:1); //*Leaky Relu Derivative
                                if(l>=2) prevDcDxsData[prevMapIndex] += reusable * (kernelData[kernelIndex]);//*kernel weight
                                sum += (prevMapData[prevMapIndex]) * reusable; //The activation of the previous layer * the correct derivative from pooling
                            }
                            thisX++;
                        }
                        thisX = 0;
                        thisY++;
                    }
                    kernelGradData[kernelIndex] += sum;
                }
            }
        }
        float biasSum = 0;
        std::vector<std::vector<bool>> done(poolDimens,std::vector<bool>(poolDimens));
        for(int y=0;y<currDimens;y++){
            int poolY = (y/poolStride);
            int currMapRow = currMapChannel + y*currMapsChildSizes[1];
            int pooledMapRow = pooledMapChannel + poolY*pooledMapsChildSizes[1];
            for(int x=0;x<currDimens;x++){
                int poolX = (x/poolStride);
                int currMapIndex = currMapRow + x;
                int pooledMapIndex = pooledMapRow + poolX;
                //Bias has to be here as otherwise it would count the same pixels multiple times
                if(floatCmp(currMapData[currMapIndex],pooledMapData[pooledMapIndex]) && !done[poolY][poolX]){
                    done[poolY][poolX] = true;
                    biasSum += (pooledDcDxsData[pooledMapIndex]) * ((currMapData[currMapIndex])<=0?0.01f:1); //Bias deriv = cost deriv * relu deriv * 1 (only 1 bias term in each new pixel expression)
                }
            }
        }
        kernelBiasesGradData[i] += biasSum;
    }
}




