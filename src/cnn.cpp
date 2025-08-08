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
    kernelsGrad = kernels; //Copies the Tensor objects by value
    weightsGrad = weights;
    resetGrad(kernelsGrad); //Don't want to apply the weights to themselves on the first iteration
    resetGrad(weightsGrad);
    this->LR = LR;
    this->activations = std::vector<Tensor>(numNeurons.size());
    for(int l=0;l<numNeurons.size();l++){
        activations[l] = Tensor({numNeurons[l]});
    }
    this->maps = std::vector<Tensor>(numMaps.size());
    for(int l=0;l<numMaps.size();l++){
        maps[l] = Tensor({numMaps[l],mapDimens[l],mapDimens[l]});
    }
    if(restart){
        resetKernels();
        resetWeights();
    }
}

//Creating a copy from a template CNN (I can't call it template)
CNN::CNN(CNN *original,float LR,Dataset *dataset,bool deepCopy) {
    numNeurons = original->numNeurons;
    numMaps = original->numMaps;
    mapDimens = original->mapDimens;
    kernelSizes = original->kernelSizes;
    strides = original->strides;
    padding = original->padding;
    d = dataset; //sharing the same dataset
    if(deepCopy){
        kernels = original->kernels; //copy by value
        weights = original->weights;
        kernelsGrad = kernels;
        weightsGrad = weights;
        resetGrad(kernelsGrad); //Don't want to apply the weights to themselves on the first iteration
        resetGrad(weightsGrad);
    }
    else{
        if(original->kernels.size()!=original->kernelsGrad.size()){
            throw std::invalid_argument("kernels and kernelsGrad must have the same number of layers");
        }
        if(original->weights.size()!=original->weightsGrad.size()){
            throw std::invalid_argument("weights and weightsGrad must have the same number of layers");
        }
        for(int i=0;i<original->kernels.size();i++){
            Tensor originalKernelLayer = original->kernels[i];
            Tensor originalKernelGradLayer = original->kernelsGrad[i];
            Tensor kernelLayer,kernelGradLayer;
            kernelLayer.shallowCopy(originalKernelLayer);
            kernelGradLayer.shallowCopy(originalKernelGradLayer);
            this->kernels.push_back(kernelLayer);
            this->kernelsGrad.push_back(kernelGradLayer);
        }
        for(int i=0;i<original->weights.size();i++){
            Tensor originalWeightLayer = original->weights[i];
            Tensor originalWeightGradLayer = original->weightsGrad[i];
            Tensor weightLayer,weightGradLayer;
            weightLayer.shallowCopy(originalWeightLayer);
            weightGradLayer.shallowCopy(originalWeightGradLayer);
            this->weights.push_back(weightLayer);
            this->weightsGrad.push_back(weightGradLayer);
        }
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
            Tensor currChannel = maps[l].slice({i}); 
            if(kernelSizes[l-1]==0){
                //1:1 mapping for a max pool layer
                Tensor prevChannel = maps[l-1].slice({i});
                currChannel = maxPool(prevChannel,strides[l-1],strides[l-1]); //maxPool requires 1:1 channels between layers
            }
            else{   
                Tensor kernel = kernels[l-1].slice({i});
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
    float *pooledData = pooled.getData().get();
    float *activations0Data = activations[0].getData().get();
    std::vector<int> pooledChildSizes = pooled.getChildSizes();
    for(int i=0;i<numMaps[numMaps.size()-1];i++){
        Tensor pooledChannel = pooled.slice({i});
        Tensor prevChannel = maps[numMaps.size()-1].slice({i});
        pooledChannel = maxPool(prevChannel,strides[strides.size()-1],strides[strides.size()-1]);
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
        float *biasesData = weights[l].getBiases()->getData().get();
        float *prevActivations = activations[l].getData().get();
        float *currActivations = activations[l+1].getData().get();
        float *currWeights = weights[l].getData().get();
        for(int i=0;i<numNeurons[l+1];i++){
            int weightsTo = i*numNeurons[l+1];
            for(int j=0;j<numNeurons[l];j++){
                //TODO likely quicker with axv2
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
    #if DEBUG >=2
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
        float *weightsGradData = weightsGrad[l].getData().get();
        float *biasesGradData = weightsGrad[l].getBiases()->getData().get();
        float *nextDcDzsData = dcDzs[l+1].getData().get();
        float *currDcDzsData = dcDzs[l].getData().get();
        float *activationsData = activations[l].getData().get();
        float *weightsData = weights[l].getData().get();
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
    int lSub1 = l-1;
    int lSub2 = l-2;
    int prevDimens = mapDimens[lSub1];
    int currDimens = mapDimens[l];
    int kernelSize = kernelSizes[lSub1];
    int kernelRadius = (int) floor(kernelSize/2);
    int thisStride = strides[lSub1];
    float *currDcDxsData = dcDxs[lSub1].getData().get(); //yes, l-1 is correct (dcDxs only has numMaps.size()-2 layers)
    float *prevDcDxsData = nullptr;
    if(l!=1) prevDcDxsData = dcDxs[lSub2].getData().get(); //No derivatives need to be stored for the first layer
    float *currMapData = maps[l].getData().get();
    float *prevMapData = maps[lSub1].getData().get();
    float *kernelData = kernels[lSub1].getData().get();
    float *kernelGradData = kernelsGrad[lSub1].getData().get(); 
    float *kernelBiasesGradData = kernelsGrad[lSub1].getBiases()->getData().get(); //only 1 for each channel (1d)
    std::vector<int> currMapsChildSizes = maps[l].getChildSizes();
    std::vector<int> prevMapsChildSizes = maps[lSub1].getChildSizes();
    std::vector<int> kernelsChildSizes = kernels[lSub1].getChildSizes();
    for(int i=0;i<numMaps[l];i++){ //For each convolution output
        int currMapChannel = i*currMapsChildSizes[0];
        int kernelToChannel = i*kernels[lSub1].getChildSizes()[0]; //kernels are [layer][nextLayerChannel][prevLayerChannel]
        for(int prevMapI=0;prevMapI<numMaps[lSub1];prevMapI++){ //For each previous channel
            int prevMapChannel = prevMapI*prevMapsChildSizes[0];
            int kernelFromChannel = kernelToChannel + prevMapI*kernelsChildSizes[1];
            for(int j=0;j<kernelSize;j++){
                int kernelRow = kernelFromChannel + j*kernelsChildSizes[2];
                for(int k=0;k<kernelSize;k++){ //For each element in the kernel (k,j)
                    //Add up all the activations that it sees
                    int kernelIndex = kernelRow + k;
                    float sum = 0;
                    int thisY,thisX;
                    thisY = thisX = 0;
                    int yStart, yEnd;
                    int xStart, xEnd;
                    if(padding){
                        yStart = (j<kernelRadius)? floorMod((j-kernelRadius),thisStride) : j-kernelRadius; //want modulus (positive) not the remainder
                        yEnd = std::min(prevDimens-kernelRadius+j,prevDimens); //When j>=kernelRadius, it reaches the end item. We don't care about the stride as this is the upper bound
                        xStart = (k<kernelRadius)? floorMod((k-kernelRadius),thisStride) : k-kernelRadius;
                        xEnd = std::min(prevDimens-kernelRadius+k,prevDimens); //Same here - makes sense with a drawing
                        //The limits are needed as we have removed the padding and so we have to stop it earlier
                    }
                    else{
                        yStart = j;
                        yEnd = prevDimens-kernelSize+j+1;
                        xStart = k;
                        xEnd = prevDimens-kernelSize+k+1;
                    }
                    for(int y=yStart;y<yEnd;y+=thisStride){  //For every pixel in the previous layer (x,y) which then corresponds to one in the current (x-k,y-j)
                        int currMapRow = currMapChannel + thisY*currMapsChildSizes[1];
                        int prevMapRow = prevMapChannel + y*prevMapsChildSizes[1];
                        for(int x=xStart;x<xEnd;x+=thisStride){
                            int currMapIndex = currMapRow + thisX;
                            int prevMapIndex = prevMapRow + x;
                            if(!floatCmp(currDcDxsData[currMapIndex],0.0f)){ 
                                float reusable = currDcDxsData[currMapIndex] //Previous derivative (from pooling)
                            *  ((currMapData[currMapIndex])<=0?0.01f:1); //*Leaky Relu Derivative
                                sum += (prevMapData[prevMapIndex]) * reusable;//The previous activation
                                if(l!=1) prevDcDxsData[prevMapIndex] += reusable * kernelData[kernelIndex]; //don't have dcDxs for the first layer
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
        //Bias doesn't care about the inputs and so only needs the output
        float biasSum = 0;
        for(int y=0;y<currDimens;y++){
            int currMapRow = currMapChannel + y*currMapsChildSizes[1];
            for(int x=0;x<currDimens;x++){
                int currMapI = currMapRow + x;
                //Bias has to be here as otherwise it would count the same pixels multiple times
                biasSum += (currDcDxsData[currMapI]) * ((currMapData[currMapI])<=0?0.01f:1); //Bias deriv = cost deriv * relu deriv * 1 (only 1 bias term in each new pixel expression)
            }            
        }
        kernelBiasesGradData[i] += biasSum;
    }
}

void CNN::finalPoolingConvBackwards(std::vector<Tensor>& dcDzs,std::vector<Tensor>& dcDxs,bool padding){
    int lastMapsL = maps.size()-1;
    int prevMapsL = maps.size()-2;
    int lastKernelsL = kernels.size()-1;
    Tensor *kernelBiasesGrad = kernelsGrad[lastKernelsL].getBiases(); //only 1 for each channel (1d)
    float *activations0Data = activations[0].getData().get();
    float *lastMapData = maps[lastMapsL].getData().get();
    float *prevMapsData = maps[prevMapsL].getData().get();
    float *dcDzs0Data = dcDzs[0].getData().get();
    float *lastDcDxsData = dcDxs[dcDxs.size()-1].getData().get();
    float *kernelData = kernels[lastKernelsL].getData().get();
    float *kernelGradData = kernelsGrad[lastKernelsL].getData().get();
    float *kernelBiasesGradData = kernelsGrad[lastKernelsL].getBiases()->getData().get(); //only 1 for each channel (1d)
    int prevDimens = mapDimens[prevMapsL];
    int currDimens = mapDimens[lastMapsL];
    int kernelSize = kernelSizes[lastKernelsL];
    int kernelRadius = (int) floor(kernelSize/2);
    int poolStride = strides[strides.size()-1];
    int thisStride = strides[strides.size()-2];
    int poolWidth = mapDimens[lastMapsL]/strides[strides.size()-1];
    int poolArea = poolWidth*poolWidth;
    std::vector<int> lastMapsChildSizes = maps[lastMapsL].getChildSizes();
    std::vector<int> lastKernelsChildSizes = kernels[lastKernelsL].getChildSizes();
    for(int i=0;i<numMaps[lastMapsL];i++){ //for each final map
        int mlpRegion = i*poolArea; 
        int lastMapChannel = i*lastMapsChildSizes[0];
        int kernelToChannel = i*lastKernelsChildSizes[0];
        for(int prevMapI=0;prevMapI<numMaps[prevMapsL];prevMapI++){
            int prevMapChannel = prevMapI*lastMapsChildSizes[0];
            int kernelFromChannel = kernelToChannel + prevMapI*lastKernelsChildSizes[1];
            for(int j=0;j<kernelSize;j++){
                int kernelRow = kernelFromChannel + j*lastKernelsChildSizes[2];
                for(int k=0;k<kernelSize;k++){ //For each element in the kernel (k,j)
                    int kernelIndex = kernelRow + k;
                    //Add up all the activations that it sees
                    float sum = 0;
                    int thisY,thisX;
                    thisY = thisX = 0;
                    std::vector<bool> done(poolArea); //don't count the max pixel more than once
                    int yStart, yEnd;
                    int xStart, xEnd;
                    if(padding){
                        yStart = (j<kernelRadius)? floorMod((j-kernelRadius),thisStride) : j-kernelRadius; //want modulus (positive) not the remainder
                        yEnd = std::min(prevDimens-kernelRadius+j,prevDimens); //When j>=kernelRadius, it reaches the end item. We don't care about the stride as this is the upper bound
                        xStart = (k<kernelRadius)? floorMod((k-kernelRadius),thisStride) : k-kernelRadius;
                        xEnd = std::min(prevDimens-kernelRadius+k,prevDimens); //Same here - makes sense with a drawing
                        //The limits are needed as we have removed the padding and so we have to stop it earlier
                    }
                    else{
                        yStart = j;
                        yEnd = prevDimens-kernelSize+j+1;
                        xStart = k;
                        xEnd = prevDimens - kernelSize+k+1;
                    }
                    for(int y=yStart;y<yEnd;y+=thisStride){  //For every pixel in the previous layer (x,y) which then corresponds to one in the current (thisX,thisY)
                        int lastMapRow = lastMapChannel + thisY*lastMapsChildSizes[1];
                        int prevMapRow = prevMapChannel + y*lastMapsChildSizes[1];
                        int mlpSection = (((thisY)/poolStride)*poolWidth);
                        for(int x=xStart;x<xEnd;x+=thisStride){ //Derivatve of the corresponding pixel in the next (backwards) layer
                            int mlpSubIndex = mlpSection + ((thisX)/poolStride);
                            int mlpIndex = mlpRegion + mlpSubIndex;
                            int lastMapIndex = lastMapRow + thisX;
                            int prevMapIndex = prevMapRow + x;
                            if(floatCmp(lastMapData[lastMapIndex],activations0Data[mlpIndex]) && !done[mlpSubIndex]){ //only the max element has a derivative
                                done[mlpSubIndex] = true;
                                //In the first MLP layer a=relu(x) where x is the max activation pixel from pooling
                                lastDcDxsData[prevMapIndex] += dcDzs0Data[mlpIndex] * kernelData[kernelIndex];//*kernel weight
                                sum+= prevMapsData[prevMapIndex] * dcDzs0Data[mlpIndex]; //The activation of the previous layer * the correct derivative from pooling
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
        std::vector<bool> done(poolArea);
        for(int y=0;y<currDimens;y++){
            int mlpSection = (((y)/poolStride)*poolWidth);
            int lastMapRow = lastMapChannel + y*lastMapsChildSizes[1];
            for(int x=0;x<currDimens;x++){
                int mlpSubIndex = mlpSection + ((x)/poolStride);
                int mlpIndex = mlpRegion + mlpSubIndex;
                int lastMapIndex = lastMapRow + x;
                //Bias has to be here as otherwise it would count the same pixels multiple times
                if(floatCmp(lastMapData[lastMapIndex],activations0Data[mlpIndex]) && !done[mlpSubIndex]){
                    done[mlpSubIndex] = true;
                    biasSum += dcDzs0Data[mlpIndex]; //Bias deriv = cost deriv * relu deriv * 1 (only 1 bias term in each new pixel expression)
                }
            }
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
    float *currMapData = maps[l].getData().get(); //None of these will have any offsets i.e. they aren't sub-tensors
    float *prevMapData = maps[lSub1].getData().get();
    float *pooledMapData = maps[lPlus1].getData().get();
    float *pooledDcDxsData = dcDxs[l].getData().get();
    float *prevDcDxsData = dcDxs[l-2].getData().get();
    float *kernelData = kernels[lSub1].getData().get();
    float *kernelGradData = kernelsGrad[lSub1].getData().get();
    float *kernelBiasesGradData = kernelsGrad[lSub1].getBiases()->getData().get();
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
                for(int k=0;k<kernelSize;k++){ //For each element in the kernel (k,j)
                    int kernelIndex = kernelRow + k;
                    //Add up all the activations that it sees
                    float sum = 0;
                    int thisY,thisX;
                    thisY = thisX = 0;
                    std::vector<std::vector<bool>> done(poolDimens,std::vector<bool>(poolDimens));
                    int yStart, yEnd;
                    int xStart, xEnd;
                    if(padding){
                        yStart = (j<kernelRadius)? floorMod((j-kernelRadius),thisStride) : j-kernelRadius; //want modulus (positive) not the remainder
                        yEnd = std::min(prevDimens-kernelRadius+j,prevDimens); //When j>=kernelRadius, it reaches the end item. We don't care about the stride as this is the upper bound
                        xStart = (k<kernelRadius)? floorMod((k-kernelRadius),thisStride) : k-kernelRadius;
                        xEnd = std::min(prevDimens-kernelRadius+k,prevDimens); //Same here - makes sense with a drawing
                        //The limits are needed as we have removed the padding and so we have to stop it earlier
                    }
                    else{
                        yStart = j;
                        yEnd = prevDimens-kernelSize+j+1;
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




