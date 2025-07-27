#include "cnn.hpp"

//----------------------------------------------------
//CONSTRUCTORS 

//Creating a fresh CNN
CNN::CNN(float LR,Dataset *dataset,bool restart){
    numNeurons = {1920,960,47};
    numMaps = {3,30,60,120};//includes the result of pooling (except final pooling)
    mapDimens = {128,64,32,16};
    kernelSizes = {5,3,3};  //0 represents a pooling layer, the last one is excluded
    strides = {2,2,2,4}; //pooling strides are included
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
CNN::CNN(CNN *original,float LR,Dataset *dataset) {
    numNeurons = original->numNeurons;
    numMaps = original->numMaps;
    mapDimens = original->mapDimens;
    kernelSizes = original->kernelSizes;
    strides = original->strides;
    padding = original->padding;
    d = dataset; //sharing the same dataset
    kernels = original->kernels;
    weights = original->weights;
    kernelsGrad = kernels;
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
}


//----------------------------------------------------
//KEY METHODS 


std::string CNN::forwards(Tensor& imageInt){
    #if DEBUG
        long startTime = getCurrTime();
    #endif
    reset();
    maps[0] = parseImg(imageInt);
    normaliseImg(maps[0],d->getPixelMeans(),d->getPixelStdDevs());
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
                currChannel = convolution(maps[l-1],kernel,strides[l-1],strides[l-1],padding);
            }
        }
    }
    //Final pooling 
    int poolingDimen = mapDimens[mapDimens.size()-1]/strides[strides.size()-1];
    int poolingArea = poolingDimen*poolingDimen;
    Tensor pooled({numMaps[numMaps.size()-1],poolingDimen,poolingDimen});
    for(int i=0;i<numMaps[numMaps.size()-1];i++){
        Tensor pooledChannel = pooled.slice({i});
        Tensor prevChannel = maps[numMaps.size()-1].slice({i});
        pooledChannel = maxPool(prevChannel,strides[strides.size()-1],strides[strides.size()-1]);
        for(int y=0;y<poolingDimen;y++){
            for(int x=0;x<poolingDimen;x++){
                *((activations[0])[i*poolingArea+y*poolingDimen+x]) = *(pooled[{i,y,x}]);
            }
        }
    }
    //MLP
    for(int l=0;l<weights.size();l++){
        Tensor *biases = weights[l].getBiases();
        for(int i=0;i<numNeurons[l+1];i++){
            for(int j=0;j<numNeurons[l];j++){
                *(activations[l+1][{i}]) += (*(activations[l][{j}])) * (*(weights[l][{i,j}])); 
            }
            *(activations[l+1][{i}]) += *((*biases)[{i}]); //add bias
            if(l!=weights.size()-1){ //We'll softmax the last layer and so relu is unnecessary
                *(activations[l+1][{i}]) = leakyRelu(*(activations[l+1][{i}]));
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
        System.out.println(Arrays.toString(activations[activations.size()-1]));
        System.out.println("Forwards took "+(System.currentTimeMillis()-startTime)+"ms");
    #endif
    return d->plantNames[result];
} 

void CNN::backwards(Tensor& imageInt,std::string answer){ //adds the gradient to its internal gradient arrays
    forwards(imageInt); //set all the activations
    //Gradients are not reset each time to enable batches
    #if DEBUG
        long mlpStart = getCurrTime();
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
        std::cout << "Backwards MLP took "+std::to_string(getCurrTime()-mlpStart)+"ms" << std::endl;
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
        long convolutionStart = getCurrTime();
    #endif
    //makes computational sense to do pooling and conv together
    finalPoolingConvBackwards(dcDzs,dcDxs,padding);
    for(int l=numMaps.size()-2;l>0;l--){ //>0 is due to the input dimens being included in numMaps and -2 as we've already done the last layer
        if(kernelSizes[l-1]==0){
            poolingConvBackwards(dcDxs, --l,padding); //prev (l-1) --conv-> curr (l) --pool-> pooled (l+1)
            //skip 1 layer as we have done it within poolingConvBackwards
        }
        else{
            convBackwards(dcDxs,l,padding); //prev (l-1) --conv-> curr (l)
        }
        
    }
    #if DEBUG
        System.out.println("Backwards Convolution took "+String.valueOf(System.currentTimeMillis()-convolutionStart)+"ms");
        System.out.println("Backwards took "+String.valueOf(System.currentTimeMillis()-mlpStart)+"ms");
    #endif
}


//----------------------------------------------------
//BACKPROPAGATION-RELATED

void CNN::mlpBackwards(std::vector<Tensor>& dcDzs){
    for(int l=weights.size()-1;l>=0;l--){
        Tensor *biasesGrad = weightsGrad[l].getBiases();
        for(int i=0;i<numNeurons[l+1];i++){
            for(int j=0;j<numNeurons[l];j++){//NOTE: Weights gradient != negative gradient
                weightsGrad[l][i][j] += (*dcDzs[l+1][i]) * (*activations[l][j]);
                //dC/dw = dC/da_i+1 * da_i+1/dz * dz/dw
                *dcDzs[l][j] +=  (*dcDzs[l+1][i]) * (*weights[l][{i,j}]) * ((*(activations[l][j])<=0)?0.01f:1);//next layer
                //dC/dz_i = dC/dz_i+1 * dz_i+1/da_i * da_i/dz_i
            }
            //bias
            *(*biasesGrad)[i] += *dcDzs[l+1][i];
        }
    }
}

void CNN::convBackwards(std::vector<Tensor>& dcDxs, int l,bool padding){
    int lSub1 = l-1;
    int lSub2 = l-2;
    Tensor *kernelBiasesGrad = kernelsGrad[lSub1].getBiases(); //only 1 for each channel (1d)
    int prevDimens = mapDimens[lSub1];
    int currDimens = mapDimens[l];
    int kernelSize = kernelSizes[lSub1];
    int kernelRadius = (int) floor(kernelSize/2);
    int thisStride = strides[lSub1];
    for(int i=0;i<numMaps[l];i++){ //For each convolution output
        for(int prevMapI=0;prevMapI<numMaps[lSub1];prevMapI++){ //For each previous channel
            for(int j=0;j<kernelSize;j++){
                for(int k=0;k<kernelSize;k++){ //For each element in the kernel (k,j)
                    //Add up all the activations that it sees
                    float sum = 0;
                    float reusable;
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
                        for(int x=xStart;x<xEnd;x+=thisStride){
                            if(!floatCmp(*dcDxs[lSub1][{i,thisY,thisX}],0.0f)){ //yes, l-1 is correct (dcDxs only has numMaps.size()-2 layers)
                                reusable = *dcDxs[lSub1][{i,thisY,thisX}] //Previous derivative (from pooling)
                            *  ((*maps[l][{i,thisY,thisX}])<=0?0.01f:1); //*Leaky Relu Derivative
                                sum += (*maps[lSub1][{prevMapI,y,x}]) * reusable;//The previous activation
                                if(l!=1) *dcDxs[lSub2][{prevMapI,y,x}] += reusable * (*kernels[lSub1][{i,prevMapI,j,k}]); //don't have dcDxs for the first layer
                            }
                            thisX++;
                        }
                        thisX = 0;
                        thisY++;
                    }
                    *kernelsGrad[lSub1][{i,prevMapI,j,k}] += sum; //kernels are [layer][nextLayerChannel][prevLayerChannel]
                }
            }
        }
        //Bias doesn't care about the inputs and so only needs the output
        float biasSum = 0;
        for(int y=0;y<currDimens;y++){
            for(int x=0;x<currDimens;x++){
                //Bias has to be here as otherwise it would count the same pixels multiple times
                biasSum += (*dcDxs[lSub1][{i,y,x}]) * ((*maps[l][{i,y,x}])<=0?0.01f:1); //Bias deriv = cost deriv * relu deriv * 1 (only 1 bias term in each new pixel expression
            }            
        }
        *(*kernelBiasesGrad)[i] += biasSum;
    }
}

void CNN::finalPoolingConvBackwards(std::vector<Tensor>& dcDzs,std::vector<Tensor>& dcDxs,bool padding){
    Tensor *kernelBiasesGrad = kernelsGrad[kernels.size()-1].getBiases(); //only 1 for each channel (1d)
    for(int i=0;i<numMaps[numMaps.size()-1];i++){ //for each final map
        int prevDimens = mapDimens[numMaps.size()-2];
        int currDimens = mapDimens[numMaps.size()-1];
        int kernelSize = kernelSizes[kernelSizes.size()-1];
        int kernelRadius = (int) floor(kernelSize/2);
        int poolStride = strides[strides.size()-1];
        int thisStride = strides[strides.size()-2];
        int poolWidth = mapDimens[numMaps.size()-1]/strides[strides.size()-1];
        int poolArea = poolWidth*poolWidth;
        int mlpRegion = i*poolArea; //Saving a few multiplications
        for(int prevMapI=0;prevMapI<numMaps[numMaps.size()-2];prevMapI++){
            for(int j=0;j<kernelSize;j++){
                for(int k=0;k<kernelSize;k++){ //For each element in the kernel (k,j)
                    //Add up all the activations that it sees
                    float sum = 0;
                    int thisY,thisX,mlpIndex,mlpSubIndex;
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
                        for(int x=xStart;x<xEnd;x+=thisStride){ //Derivatve of the corresponding pixel in the next (backwards) layer
                            mlpSubIndex = (((thisY)/poolStride)*poolWidth) + ((thisX)/poolStride);
                            mlpIndex = mlpRegion + mlpSubIndex;
                            if(floatCmp(*maps[maps.size()-1][{i,thisY,thisX}],*activations[0][mlpIndex]) && !done[mlpSubIndex]){ //only the max element has a derivative
                                done[mlpSubIndex] = true;
                                //In the first MLP layer a=relu(x) where x is the max activation pixel from pooling
                                *dcDxs[dcDxs.size()-1][{prevMapI,y,x}] += (*dcDzs[0][mlpIndex]) * (*kernels[kernels.size()-1][{i,prevMapI,j,k}]);//*kernel weight
                                sum+= (*maps[maps.size()-2][{prevMapI,y,x}]) * (*dcDzs[0][mlpIndex]); //The activation of the previous layer * the correct derivative from pooling
                            }
                            thisX++;
                        }
                        thisX = 0;
                        thisY++;
                    }
                    *kernelsGrad[kernels.size()-1][{i,prevMapI,j,k}] += sum;
                }
            }
        }
        float biasSum = 0;
        int mlpIndex,mlpSubIndex;
        std::vector<bool> done(poolArea);
        for(int y=0;y<currDimens;y++){
            for(int x=0;x<currDimens;x++){
                mlpSubIndex = (((y)/poolStride)*poolWidth) + ((x)/poolStride);
                mlpIndex = mlpRegion + mlpSubIndex;
                //Bias has to be here as otherwise it would count the same pixels multiple times
                if(floatCmp(*maps[maps.size()-1][{i,y,x}],*activations[0][mlpIndex]) && !done[mlpSubIndex]){
                    done[mlpSubIndex] = true;
                    biasSum += *dcDzs[0][mlpIndex]; //Bias deriv = cost deriv * relu deriv * 1 (only 1 bias term in each new pixel expression
                }
            }
        }
        *(*kernelBiasesGrad)[i] += biasSum;
    }
}

void CNN::poolingConvBackwards(std::vector<Tensor>& dcDxs, int l,bool padding){
    for(int i=0;i<numMaps[l];i++){ //prev (l-1) --conv-> curr (l) --pool-> pooled (l+1)
        //pooling is 1:1 between channels
        int prevDimens = mapDimens[l-1];
        int currDimens = mapDimens[l];
        int kernelSize = kernelSizes[l-1];
        int kernelRadius = (int) floor(kernelSize/2);
        int poolStride = strides[l];
        int thisStride = strides[l-1];
        int poolDimens = mapDimens[l+1];
        for(int prevMapI=0;prevMapI<numMaps[l-1];prevMapI++){
            for(int j=0;j<kernelSize;j++){
                for(int k=0;k<kernelSize;k++){ //For each element in the kernel (k,j)
                    //Add up all the activations that it sees
                    float sum = 0;
                    float reusable;
                    int thisY,thisX,poolX,poolY;
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
                        for(int x=xStart;x<xEnd;x+=thisStride){ //Derivatve of the corresponding pixel in the next (backwards) layer
                            poolY = ((thisY)/poolStride);
                            poolX = ((thisX)/poolStride);
                            if(floatCmp(*maps[l][{i,thisY,thisX}],*maps[l+1][{i,poolY,poolX}]) && !done[poolY][poolX] && !floatCmp(*dcDxs[l][{i,poolY,poolX}],0.0f)){ //only the max element has a derivative
                                done[poolY][poolX] = true;
                                reusable =  *dcDxs[l][{i,poolY,poolX}]//Previous derivative (from pooling)
                                * ((*maps[l][{i,thisY,thisX}])<=0?0.01f:1); //*Leaky Relu Derivative
                                if(l!=1) *dcDxs[l-2][{i,y,x}] += reusable * (*kernels[l-1][{i,prevMapI,j,k}]);//*kernel weight
                                sum += (*maps[l-1][{prevMapI,y,x}]) * reusable; //The activation of the previous layer * the correct derivative from pooling
                            }
                            thisX++;
                        }
                        thisX = 0;
                        thisY++;
                    }
                    *kernelsGrad[l-1][{i,prevMapI,j,k}] += sum;
                }
            }
        }
        float biasSum = 0;
        int poolX,poolY;
        std::vector<std::vector<bool>> done(poolDimens,std::vector<bool>(poolDimens));
        for(int y=0;y<currDimens;y++){
            for(int x=0;x<currDimens;x++){
                poolY = (y/poolStride);
                poolX = (x/poolStride);
                //Bias has to be here as otherwise it would count the same pixels multiple times
                if(floatCmp(*maps[l][{i,y,x}],*maps[l+1][{i,poolY,poolX}]) && !done[poolY][poolX]){
                    done[poolY][poolX] = true;
                    biasSum += (*dcDxs[l][{i,poolY,poolX}]) * ((*(maps[l][{i,y,x}]))<=0?0.01f:1); //Bias deriv = cost deriv * relu deriv * 1 (only 1 bias term in each new pixel expression)
                }
            }
        }
        *kernelsGrad[l-1][{i,(numMaps[l-1]-1),(kernelSize-1),kernelSize}] += biasSum;
    }
}




