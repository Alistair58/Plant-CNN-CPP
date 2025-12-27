#include "cnn.hpp"
#if PROFILING
    #include "timer.hpp"
#endif

//----------------------------------------------------
//CONSTRUCTORS 

//Creating a fresh CNN
CNN::CNN(float LR,Dataset *dataset,bool restart,float dropoutProbability){
    //Model 5:
    // 128x128x3
    // 64x64x30 (3x3 conv stride 2)
    // 32x32x60 (3x3 conv stride 2)
    // 16x16x120 (3x3 conv stride 2)
    // 4x4x120 (max pool)
    // 1920
    // 1920 (FC)
    // 47 (FC)
    numNeurons = {1920,1920,47};
    //includes the result of pooling (except final pooling)
    mapDimens = std::vector<dimens>(4);
    mapDimens[0] = {3,128,128};
    mapDimens[1] = {30,64,64};
    mapDimens[2] = {60,32,32};
    mapDimens[3] = {120,16,16};
    //0 represents a pooling layer, the last one is excluded
    kernelSizes = std::vector<std::pair<int,int>>(3);
    kernelSizes[0] = {3,3}; //h,w
    kernelSizes[1] = {3,3};
    kernelSizes[2] = {3,3};
    //pooling strides are included
    strides = std::vector<std::pair<int,int>>(4);
    strides[0] = {2,2};//y,x - pooling strides are included
    strides[1] = {2,2};
    strides[2] = {2,2};
    strides[3] = {4,4};
    padding = true;
    this->dropoutProb = dropoutProbability;

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
    this->maps = std::vector<Tensor>(mapDimens.size());
    for(int l=0;l<mapDimens.size();l++){
        maps[l] = Tensor({mapDimens[l].c,mapDimens[l].h,mapDimens[l].w});
    }
    if(padding){
        this->paddedMaps = std::vector<Tensor>(mapDimens.size()-1); //last map is pooled not convolved - my favourite way of having a Martini
        for(int l=0;l<mapDimens.size()-1;l++){
            int kernelRadiusY = std::floor(this->kernelSizes[l].first/2);
            int kernelRadiusX = std::floor(this->kernelSizes[l].second/2);
            int paddedHeight = mapDimens[l].h+2*kernelRadiusY;
            int paddedWidth = mapDimens[l].w+2*kernelRadiusX;
            paddedMaps[l] = Tensor({mapDimens[l].c,paddedHeight,paddedWidth});
        }
    }
    for(int l=0;l<kernelSizes.size();l++){
        if(kernelSizes[l].first==0 || kernelSizes[l].second==0){ //pooling
            int pooledDimenX = mapDimens[l].w/strides[l].second;
            int pooledDimenY = mapDimens[l].h/strides[l].first;
            maxPoolIndices.push_back(std::unique_ptr<int[]>(new int[mapDimens[l].c*pooledDimenY*pooledDimenX]));
        }
    }
    //final pooling
    int finalPooledDimenX = mapDimens[mapDimens.size()-1].w/strides[strides.size()-1].second;
    int finalPooledDimenY = mapDimens[mapDimens.size()-1].h/strides[strides.size()-1].first;
    maxPoolIndices.push_back(std::unique_ptr<int[]>(
        new int[mapDimens[mapDimens.size()-1].c*finalPooledDimenY*finalPooledDimenX]
    ));
    if(restart){
        resetKernels();
        resetWeights();
    }
}

//Creating a copy from a template (original) CNN
CNN::CNN(CNN *original,float LR,Dataset *dataset,bool deepCopyWeights) {
    numNeurons = original->numNeurons;
    mapDimens = original->mapDimens;
    kernelSizes = original->kernelSizes;
    strides = original->strides;
    padding = original->padding;
    dropoutProb = original->dropoutProb;
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
    this->maps = std::vector<Tensor>(mapDimens.size());
    for(int l=0;l<mapDimens.size();l++){
        maps[l] = Tensor({mapDimens[l].c,mapDimens[l].h,mapDimens[l].w});
    }
    if(padding){
        this->paddedMaps = std::vector<Tensor>(mapDimens.size()-1); //last map is pooled not convolved - my favourite way of having a Martini
        for(int l=0;l<mapDimens.size()-1;l++){
            int kernelRadiusY = std::floor(this->kernelSizes[l].first/2);
            int kernelRadiusX = std::floor(this->kernelSizes[l].second/2);
            int paddedHeight = mapDimens[l].h+2*kernelRadiusY;
            int paddedWidth = mapDimens[l].w+2*kernelRadiusX;
            paddedMaps[l] = Tensor({mapDimens[l].c,paddedHeight,paddedWidth});
        }
    }
    for(int l=0;l<kernelSizes.size();l++){
        if(kernelSizes[l].first==0 || kernelSizes[l].second==0){ //pooling
            int pooledDimenX = mapDimens[l].w/strides[l].second;
            int pooledDimenY = mapDimens[l].h/strides[l].first;
            maxPoolIndices.push_back(std::unique_ptr<int[]>(new int[mapDimens[l].c*pooledDimenY*pooledDimenX]));
        }
    }
    //final pooling
    int finalPooledDimenX = mapDimens[mapDimens.size()-1].w/strides[strides.size()-1].second;
    int finalPooledDimenY = mapDimens[mapDimens.size()-1].h/strides[strides.size()-1].first;
    maxPoolIndices.push_back(std::unique_ptr<int[]>(
        new int[mapDimens[mapDimens.size()-1].c*finalPooledDimenY*finalPooledDimenX]
    ));
}


//----------------------------------------------------
//KEY METHODS 


std::string CNN::forwards(Tensor& imageInt,bool training
#if PROFILING
    ,Timer *parentTimer
#endif
){
    #if PROFILING
        Timer *forwardsTimer = parentTimer->addChildTimer("forwards");
    #endif
    //reset all the values in maps and activations
    reset();

    //Downsize the input such that it fits in the first layer
    maps[0] = parseImg(imageInt
    #if PROFILING
        ,parentTimer?forwardsTimer:nullptr
    #endif
    );

    normaliseImg(maps[0],d->getPixelMeans(),d->getPixelStdDevs()
    #if PROFILING
        ,parentTimer?forwardsTimer:nullptr
    #endif
    );
    #if PROFILING
        Timer *convolutionalLayersTimer = nullptr;
        if(parentTimer) convolutionalLayersTimer = forwardsTimer->addChildTimer("convolutionalLayers");
    #endif
    //Convolutional and pooling layers
    for(int l=1;l<mapDimens.size();l++){
        #if PROFILING
            Timer *convolutionalLayerTimer = nullptr;
            if(parentTimer) convolutionalLayerTimer = convolutionalLayersTimer->addChildTimer("convolutionLayer"+std::to_string(l-1));
        #endif
        for(int i=0;i<mapDimens[l].c;i++){
            //Does copy-elision and so no ctor is called and memory is shared
            Tensor currChannel = maps[l].slice({i}); 
            if(kernelSizes[l-1].first==0 || kernelSizes[l-1].second==0){
                //1:1 mapping for a max pool layer
                Tensor prevChannel = maps[l-1].slice({i});
                currChannel = maxPool(prevChannel,strides[l-1].second,strides[l-1].first); //maxPool requires 1:1 channels between layers
            }
            else{   
                //Slice with biases
                Tensor kernel = kernels[l-1].slice({i},{i});
                if(i==0){
                    //We need to set the paddedMap with the correct data and padding 
                    currChannel = convolution(maps[l-1],paddedMaps[l-1],kernel,strides[l-1].second,strides[l-1].first
                    #if PROFILING
                        ,parentTimer?convolutionalLayerTimer:nullptr
                    #endif
                    );
                }   
                else{
                    //We've already set the paddedMap with the correct data
                    //Tell it not to pad
                    currChannel = convolution(paddedMaps[l-1],kernel,strides[l-1].second,strides[l-1].first,false
                    #if PROFILING
                        ,parentTimer?convolutionalLayerTimer:nullptr
                    #endif
                    );
                }
            }
        }
        #if PROFILING
            if(parentTimer) convolutionalLayerTimer->stop();
        #endif
    }
    #if PROFILING
        if(parentTimer) convolutionalLayersTimer->stop();
    #endif

    finalPooling(
    #if PROFILING
        parentTimer?forwardsTimer:nullptr
    #endif
    );
    
    mlpForwards(training
    #if PROFILING
        ,parentTimer?forwardsTimer:nullptr
    #endif
    );
    
    float largestActivation = *(activations[activations.size()-1][0]);
    int result = 0;
    for(int i=1;i<numNeurons[numNeurons.size()-1];i++){
        if(*(activations[activations.size()-1][i])>largestActivation){
            largestActivation = *(activations[activations.size()-1][i]);
            result = i;
        }
    }

    #if DEBUG
        d1 outputVec = activations[activations.size()-1].toVector<d1>();
        std::cout << "[";
        for(int i=0;i<outputVec.size()-1;i++){
            std::cout << std::to_string(outputVec[i])+",";
        }
        std::cout << std::to_string(outputVec[outputVec.size()-1])+"]" << std::endl;
    #endif

    #if DEBUG >= 2
        saveMaps();
        saveActivations();
    #endif

    #if PROFILING
        if(parentTimer) forwardsTimer->stop();
    #endif

    return d->plantNames[result];
} 

void CNN::backwards(Tensor& imageInt,std::string answer
#if PROFILING
    ,Timer *parentTimer
#endif
){ 
    //Adds the gradient to its internal gradient arrays
    #if PROFILING
        Timer *backwardsTimer = nullptr;
        if(parentTimer) backwardsTimer = parentTimer->addChildTimer("backwards");
    #endif

    //Set all the activations
    forwards(imageInt,true
    #if PROFILING
        ,parentTimer?backwardsTimer:nullptr
    #endif
    ); 

    //Gradients are not reset each time to enable batches
    #if PROFILING
        Timer *mlpTimer = nullptr;
        if(parentTimer) mlpTimer = backwardsTimer->addChildTimer("mlp");
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
        dcDzs[l] = Tensor({numNeurons[l]}); //All layers need activation derivatives
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
    #if PROFILING
        Timer *finalPoolingConvTimer = nullptr;
        if(parentTimer){
            mlpTimer->stop();
            finalPoolingConvTimer = backwardsTimer->addChildTimer("finalPoolingConv");
        }
    #endif
    //x is the image pixel value and so these dcDxs are the derivatives based on pixels which are carried backwards
    std::vector<Tensor> dcDxs(mapDimens.size()-2);
    //No dcDxs for first or last (last goes straight into the MLP)
    for(int l=0;l<mapDimens.size()-2;l++){
        if(kernelSizes[l+1].first==0 || kernelSizes[l+1].second==0){
            //There doesn't need to be any dcDxs for any pre-pooling maps
            //Blank dcDxs for pre-pooling layers so that indices stay consistent with map indices
            dcDxs[l] = Tensor({0});
        }
        else{
            dcDxs[l] = Tensor({mapDimens[l+1].c,mapDimens[l+1].h,mapDimens[l+1].w});
        }
    }
    //makes computational sense to do pooling and conv together
    finalPoolingConvBackwards(dcDzs,dcDxs,padding);
    #if PROFILING
        Timer *convolutionsTimer = nullptr;
        if(parentTimer){
            finalPoolingConvTimer->stop();
            convolutionsTimer = backwardsTimer->addChildTimer("convolutions");
        }
    #endif
    for(int l=mapDimens.size()-2;l>0;l--){ //>0 is due to the input dimens being included in numMaps and -2 as we've already done the last layer
        #if PROFILING
            Timer *convolutionsLayerTimer = nullptr;
            if(parentTimer){
                convolutionsLayerTimer = convolutionsTimer->addChildTimer("convolutionsLayer"+std::to_string(l-1)); 
            }
        #endif
        if(kernelSizes[l-1].first==0 || kernelSizes[l-1].second==0){
            poolingConvBackwards(dcDxs, --l,padding); //prev (l-1) --conv-> curr (l) --pool-> pooled (l+1)
            //skip 1 layer as we have done it within poolingConvBackwards
            #if PROFILING
                if(parentTimer) convolutionsLayerTimer->setNote("(pooling)");
            #endif
        }
        else{
            //prev (l-1) --conv-> curr (l)
            convBackwards(dcDxs,l,padding
            #if PROFILING
                ,convolutionsLayerTimer
            #endif
            ); 
        }
        #if PROFILING
            if(parentTimer) convolutionsLayerTimer->stop();
        #endif
        
    }
    #if PROFILING
        if(parentTimer){
            convolutionsTimer->stop();
            backwardsTimer->stop();
        }
    #endif
}

//----------------------------------------------------
//FORWARDS-RELATED

void CNN::finalPooling(
#if PROFILING
    Timer *parentTimer
#endif
){
    #if PROFILING
        Timer *finalPoolingTimer = nullptr;
        if(parentTimer) finalPoolingTimer = parentTimer->addChildTimer("finalPooling");
    #endif

    //Max pool the elements in the final convolutional layer and 
    // set the values in the first MLP layer to the result of this
    int poolingDimenX = mapDimens[mapDimens.size()-1].w/strides[strides.size()-1].second;
    int poolingDimenY = mapDimens[mapDimens.size()-1].h/strides[strides.size()-1].first;
    int poolingArea = poolingDimenY*poolingDimenX;
    //Temporary result store - maxPool returns a Tensor
    Tensor pooled({mapDimens[mapDimens.size()-1].c,poolingDimenY,poolingDimenX});
    float *pooledData = pooled.getData();
    float *activations0Data = activations[0].getData();
    std::vector<int> pooledChildSizes = pooled.getChildSizes();
    for(int i=0;i<mapDimens[mapDimens.size()-1].c;i++){
        Tensor pooledChannel = pooled.slice({i});
        Tensor prevChannel = maps[mapDimens.size()-1].slice({i});
        int *maxPoolIndicesMap = &(maxPoolIndices[maxPoolIndices.size()-1][i*poolingArea]);
        pooledChannel = maxPool(prevChannel,strides[strides.size()-1].second,strides[strides.size()-1].first,maxPoolIndicesMap);
        int activationsPoolingArea = i*poolingArea;
        int poolingChannel = i*pooledChildSizes[0];
        for(int y=0;y<poolingDimenY;y++){
            int activationsPoolingRow = activationsPoolingArea + y*poolingDimenX;
            int poolingRow = poolingChannel + y*pooledChildSizes[1];
            //memcpy can be vectorised
            std::memcpy(
                activations0Data+activationsPoolingRow,
                pooledData+poolingRow,
                poolingDimenX*sizeof(float)
            );
        }
    }

    #if PROFILING
        if(parentTimer) finalPoolingTimer->stop();
    #endif
}

void CNN::mlpForwards(bool training
#if PROFILING
    ,Timer *parentTimer
#endif
){
    #if PROFILING
        Timer *mlpForwardsTimer = nullptr;
        if(parentTimer) mlpForwardsTimer = parentTimer->addChildTimer("mlpForwards");
    #endif

    std::uniform_real_distribution dropoutDist(0.0f,1.0f);

    for(int l=0;l<weights.size();l++){
        float *biasesData = weights[l].getBiases()->getData();
        float* __restrict__ prevActivations = activations[l].getData();
        float* __restrict__ currActivations = activations[l+1].getData();
        float *currWeights = weights[l].getData();
        for(int i=0;i<numNeurons[l+1];i++){
            //Dropout
            if(training && l!=weights.size()-1 && dropoutDist(localRng)<=dropoutProb) continue; 
            int weightsTo = i*numNeurons[l];
            int j=0;
            float *currWeightsTo = currWeights + weightsTo;
            //Process 8 previous layer activations and their corresponding weights simultaneously
            for(;j+7<numNeurons[l];j+=8){
                __m256 prevActivationsM256 = _mm256_loadu_ps(&prevActivations[j]);
                __m256 currWeightsM256 = _mm256_loadu_ps(&currWeightsTo[j]);
                currActivations[i] += dotProduct8f(prevActivationsM256,currWeightsM256);
            }
            //scalar tail
            for(;j<numNeurons[l];j++){
                currActivations[i] += prevActivations[j] * currWeightsTo[j]; 
            }
            currActivations[i] += biasesData[i]; //add bias
            if(l!=weights.size()-1){ //We'll softmax the last layer and so relu is unnecessary
                currActivations[i]= leakyRelu(currActivations[i]);
            }
        }
    }
    activations[activations.size()-1] = softmax(activations[activations.size()-1].toVector<d1>());

    #if PROFILING
        if(parentTimer) mlpForwardsTimer->stop();
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
        float* __restrict__ daDzs = (float*) malloc(activations[l].getTotalSize()*sizeof(float));
        if(!daDzs){
            throw std::runtime_error("Failed malloc in mlpBackwards");
        }
        for(int j=0;j<activations[l].getTotalSize();j++){
            daDzs[j] = (((activationsData[j])<=0)?0.01f:1);
        }
        for(int i=0;i<numNeurons[l+1];i++){
            int weightsNeuron = i*numNeurons[l];
            float* weightsToData = weightsData+weightsNeuron;
            float* weightsToGradData = weightsGradData+weightsNeuron;
            //NOTE: Weights gradient != negative gradient
            int j=0;
            __m256 nextDcDzsM256 = _mm256_set1_ps(nextDcDzsData[i]);
            for(;j+7<numNeurons[l];j+=8){
                __m256 weightsGradM256 = _mm256_loadu_ps(&weightsToGradData[j]);
                __m256 activationsM256 = _mm256_loadu_ps(&activationsData[j]);
                weightsGradM256 = _mm256_fmadd_ps(nextDcDzsM256,activationsM256,weightsGradM256);
                _mm256_storeu_ps(&weightsToGradData[j],weightsGradM256);
                //dC/dw = dC/da_i+1 * da_i+1/dz * dz/dw
                __m256 currDcDzsM256 = _mm256_loadu_ps(&currDcDzsData[j]);
                __m256 weightsM256 = _mm256_loadu_ps(&weightsToData[j]); 
                __m256 daDzsM256 = _mm256_loadu_ps(&daDzs[j]);
                __m256 acc = _mm256_mul_ps(nextDcDzsM256,weightsM256);
                currDcDzsM256 = _mm256_fmadd_ps(acc,daDzsM256,currDcDzsM256);
                _mm256_storeu_ps(&currDcDzsData[j],currDcDzsM256);
                //dC/dz_i = dC/dz_i+1 * dz_i+1/da_i * da_i/dz_i                
            }
            //scalar tail
            for(;j<numNeurons[l];j++){
                weightsToGradData[j] += (nextDcDzsData[i]) * (activationsData[j]);
                //dC/dw = dC/da_i+1 * da_i+1/dz * dz/dw
                currDcDzsData[j] +=  (nextDcDzsData[i]) * (weightsToData[j]) * daDzs[j];//next layer
                //dC/dz_i = dC/dz_i+1 * dz_i+1/da_i * da_i/dz_i
            }
            //bias
            biasesGradData[i] += nextDcDzsData[i];
        }
        free(daDzs);
    }
}

void CNN::convBackwards(std::vector<Tensor>& dcDxs,const int l,bool padding
#if PROFILING
    ,Timer *parentTimer
#endif
){
    //We are working from the back -> front 
    //Prev is the thing closer to the input image and curr is closer the output vector
    //z = sum(k*x)+b
    //x = ReLU(z)
    const int lSub1 = l-1;
    const int lSub2 = l-2;
    const int prevDimensX = mapDimens[lSub1].w;
    const int prevDimensY = mapDimens[lSub1].h;
    const int kernelSizeX = kernelSizes[lSub1].second;
    const int kernelSizeY = kernelSizes[lSub1].first;
    const int kernelRadiusX = kernelSizeX/2;
    const int kernelRadiusY = kernelSizeY/2;
    const int thisStrideX = strides[lSub1].second;
    const int thisStrideY = strides[lSub1].first;
    float*  __restrict__ currDcDxsData = dcDxs[lSub1].getData(); //yes, l-1 is correct (dcDxs only has mapDimens.size()-2 layers)
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
    const int prevMapsChildSizes1 = prevMapsChildSizes[1];
    const int currMapsChildSizes1 = currMapsChildSizes[1];
    const int thisStrideX7 = 7*thisStrideX;
    const int thisStrideX8 = 8*thisStrideX;
    //Precompute ReLU derivatives so we don't recompute them multiple times
    const size_t currMapSize = maps[l].getTotalSize();
    #if PROFILING
        Timer *dcDzsMallocTimer = nullptr;
        if(parentTimer){
            dcDzsMallocTimer = parentTimer->addChildTimer("dcDzsMalloc");
        }
    #endif
    float* __restrict__ currDcDzsData = (float*) malloc(currMapSize*sizeof(float));
    #if PROFILING
        Timer *dcDzsLoopTimer = nullptr;
        if(parentTimer){
            dcDzsMallocTimer->stop();
            dcDzsLoopTimer = parentTimer->addChildTimer("dcDzsLoop");
        }
    #endif
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
    #if PROFILING
        Timer *preparingStridingTimer = nullptr;
        if(parentTimer){
            dcDzsLoopTimer->stop();
            preparingStridingTimer = parentTimer->addChildTimer("preparingStriding");
        }
    #endif
    //Store the strided data contiguously so it can be loaded quicker
    Tensor stridedPrevMap({mapDimens[lSub1].c,thisStrideX,mapDimens[lSub1].h,(int)std::ceil((float)mapDimens[lSub1].w/thisStrideX)});
    float* __restrict__ stridedPrevMapData = stridedPrevMap.getData();
    std::vector<int> stridedPrevMapChildSizes = stridedPrevMap.getChildSizes();
    for(int c=0;c<mapDimens[lSub1].c;c++){
        float* __restrict__ stridedPrevMapChannel = stridedPrevMapData+c*stridedPrevMapChildSizes[0];
        float* __restrict__ prevMapChannel = prevMapData+c*prevMapsChildSizes[0];
        for(int i=0;i<thisStrideX;i++){
            float* __restrict__ stridedPrevMapStride = stridedPrevMapChannel+i*stridedPrevMapChildSizes[1];
            for(int y=0;y<mapDimens[lSub1].h;y++){
                float* __restrict__ stridedPrevMapRowPtr = stridedPrevMapStride+y*stridedPrevMapChildSizes[2];
                //+i is important
                float* __restrict__ prevMapRowPtr = prevMapChannel+y*prevMapsChildSizes1 + i;
                float* __restrict__ prevMapRowEndPtr = prevMapRowPtr+prevMapsChildSizes1;

                for(;prevMapRowPtr<prevMapRowEndPtr;prevMapRowPtr+=thisStrideX,stridedPrevMapRowPtr++){
                    *stridedPrevMapRowPtr = *prevMapRowPtr;
                }
            }
        }
    }

    if(l!=1){
        //l==1 does not have dcDxs
        //Create a way for the dcDxs to be stored continguously as this is faster
        //They are put in the correct place at the end
        Tensor stridedPrevDcDxs({mapDimens[lSub1].c,thisStrideX,mapDimens[lSub1].h,(int)std::ceil((float)mapDimens[lSub1].w/thisStrideX)});
        float* __restrict__ stridedPrevDcDxsData = stridedPrevDcDxs.getData();
        std::vector<int> stridedPrevDcDxsChildSizes = stridedPrevDcDxs.getChildSizes();
        #if PROFILING
            Timer *backwardsConvLoopTimer = nullptr;
            if(parentTimer){
                preparingStridingTimer->stop();
                backwardsConvLoopTimer = parentTimer->addChildTimer("backwardsConvLoop");
            }
        #endif
        
        for(int i=0;i<mapDimens[l].c;i++){ //For each convolution output
            int currMapChannel = i*currMapsChildSizes[0];
            int kernelToChannel = i*kernelsChildSizes[0]; //kernels are [layer][nextLayerChannel][prevLayerChannel]
            for(int prevMapI=0;prevMapI<mapDimens[lSub1].c;prevMapI++){ //For each previous channel
                int prevMapChannel = prevMapI*prevMapsChildSizes[0];
                int kernelFromChannel = kernelToChannel + prevMapI*kernelsChildSizes[1];
                float* __restrict__ stridedPrevDcDxsChannel = stridedPrevDcDxsData+prevMapI*stridedPrevDcDxsChildSizes[0];
                float* __restrict__ stridedPrevMapChannel = stridedPrevMapData+prevMapI*stridedPrevMapChildSizes[0];
                for(int j=0;j<kernelSizeY;j++){
                    int kernelRow = kernelFromChannel + j*kernelsChildSizes[2];
                    int yStart, yEnd;
                    if(padding){
                        yStart = (j<kernelRadiusY)? floorMod((j-kernelRadiusY),thisStrideY) : j-kernelRadiusY; //want modulus (positive) not the remainder
                        yEnd = std::min(prevDimensY-kernelRadiusY+j,prevDimensY); //When j>=kernelRadius, it reaches the end item. We don't care about the stride as this is the upper bound
                    }
                    else{
                        yStart = j;
                        yEnd = prevDimensY-kernelSizeY+j+1;
                    }
                    for(int k=0;k<kernelSizeX;k++){ //For each element in the kernel (k,j)
                        //Add up all the activations that it sees
                        int kernelIndex = kernelRow + k;
                        float kernelVal = kernelData[kernelIndex];
                        float sum = 0;
                        int thisY,thisX;
                        thisY = thisX = 0;
                        
                        int xStart, xEnd;
                        if(padding){
                            xStart = (k<kernelRadiusX)? floorMod((k-kernelRadiusX),thisStrideX) : k-kernelRadiusX;
                            xEnd = std::min(prevDimensX-kernelRadiusX+k,prevDimensX); //Same here - makes sense with a drawing
                            //The limits are needed as we have removed the padding and so we have to stop it earlier
                        }
                        else{
                            xStart = k;
                            xEnd = prevDimensX-kernelSizeX+k+1;
                        }
                        int strideOffset = xStart%thisStrideX;
                        int xHeadstart = xStart/thisStrideX;
                        float* __restrict__ stridedPrevDcDxsOffset = stridedPrevDcDxsChannel+strideOffset*stridedPrevDcDxsChildSizes[1]+xHeadstart;
                        float* __restrict__ stridedPrevMapOffset = stridedPrevMapChannel+strideOffset*stridedPrevMapChildSizes[1]+xHeadstart;
                        for(int y=yStart;y<yEnd;y+=thisStrideY){  //For every pixel in the previous layer (x,y) which then corresponds to one in the current (x-k,y-j)
                            int currMapRow = currMapChannel + thisY*currMapsChildSizes1;
                            int prevMapRow = prevMapChannel + y*prevMapsChildSizes1;
                            float* __restrict__ currMapDcDzsRowBase = currDcDzsData+currMapRow;
                            float* __restrict__ stridedPrevDcDxsRow = stridedPrevDcDxsOffset+y*stridedPrevDcDxsChildSizes[2];
                            float* __restrict__ stridedPrevMapRow = stridedPrevMapOffset+y*stridedPrevMapChildSizes[2];
                            int x=xStart;
                            __m256 acc = _mm256_set1_ps(0);
                            for(;x+thisStrideX7<xEnd;x+=thisStrideX8){
                                float* __restrict__ currMapDcDzsBasePtr = currMapDcDzsRowBase + thisX;
                                const __m256 prevMapVals = _mm256_loadu_ps(stridedPrevMapRow+thisX);      
                                const __m256 currMapDerivs = _mm256_loadu_ps(currMapDcDzsBasePtr);
                                
                                //Add it (dC/dx*dx/dk) to kernel derivative
                                acc = _mm256_fmadd_ps(prevMapVals,currMapDerivs,acc);
                                
                                float* __restrict__ prevDcDxsPtr = stridedPrevDcDxsRow+thisX;
                                __m256 kernelVals = _mm256_set1_ps(kernelVal);
                                __m256 storedSum = _mm256_loadu_ps(prevDcDxsPtr);
                                __m256 result = _mm256_fmadd_ps(currMapDerivs,kernelVals,storedSum);
                                _mm256_storeu_ps(prevDcDxsPtr,result);
                                thisX+=8;
                            }
                            sum += horizontalSum(acc);
                            //scalar tail
                            for(;x<xEnd;x+=thisStrideX){
                                const int currMapIndex = currMapRow + thisX;
                                const int prevMapIndex = prevMapRow + x;
                                const float reusable = currDcDzsData[currMapIndex];
                                sum += (prevMapData[prevMapIndex]) * reusable;//The previous activation
                                *(stridedPrevDcDxsRow+thisX) += reusable * kernelVal; //don't have dcDxs for the first layer
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
        #if PROFILING
            Timer *unstridingDcDxsTimer = nullptr;
            if(parentTimer){
                backwardsConvLoopTimer->stop();
                unstridingDcDxsTimer = parentTimer->addChildTimer("unstridingDcDxs");
            }
        #endif
        for(int c=0;c<mapDimens[lSub1].c;c++){
            float* __restrict__ stridedPrevDcDxsChannel = stridedPrevDcDxsData+c*stridedPrevDcDxsChildSizes[0];
            float* __restrict__ prevDcDxsChannel = prevDcDxsData+c*prevMapsChildSizes[0];
            for(int i=0;i<thisStrideX;i++){
                float* __restrict__ stridedPrevDcDxsStride = stridedPrevDcDxsChannel+i*stridedPrevDcDxsChildSizes[1];
                for(int y=0;y<mapDimens[lSub1].h;y++){
                    float* __restrict__ stridedPrevDcDxsRowPtr = stridedPrevDcDxsStride+y*stridedPrevDcDxsChildSizes[2];
                    //+i is important
                    float* __restrict__ prevDcDxsRowPtr = prevDcDxsChannel+y*prevMapsChildSizes1 + i;
                    float* __restrict__ prevDcDxsRowEndPtr = prevDcDxsRowPtr+prevMapsChildSizes1;

                    for(;prevDcDxsRowPtr<prevDcDxsRowEndPtr;prevDcDxsRowPtr+=thisStrideX,stridedPrevDcDxsRowPtr++){
                        *prevDcDxsRowPtr += *stridedPrevDcDxsRowPtr;
                    }
                }
            }        
        }
        #if PROFILING
            if(parentTimer){
                unstridingDcDxsTimer->stop();
            }
        #endif
    }
    else{
        #if PROFILING
            Timer *backwardsConvLoopTimer = nullptr;
            if(parentTimer){
                preparingStridingTimer->stop();
                backwardsConvLoopTimer = parentTimer->addChildTimer("backwardsConvLoop");
            }
        #endif
        
        for(int i=0;i<mapDimens[l].c;i++){ //For each convolution output
            int currMapChannel = i*currMapsChildSizes[0];
            int kernelToChannel = i*kernelsChildSizes[0]; //kernels are [layer][nextLayerChannel][prevLayerChannel]
            for(int prevMapI=0;prevMapI<mapDimens[lSub1].c;prevMapI++){ //For each previous channel
                int prevMapChannel = prevMapI*prevMapsChildSizes[0];
                int kernelFromChannel = kernelToChannel + prevMapI*kernelsChildSizes[1];
                float* __restrict__ stridedPrevMapChannel = stridedPrevMapData+prevMapI*stridedPrevMapChildSizes[0];
                for(int j=0;j<kernelSizeY;j++){
                    int kernelRow = kernelFromChannel + j*kernelsChildSizes[2];
                    int yStart, yEnd;
                    if(padding){
                        yStart = (j<kernelRadiusY)? floorMod((j-kernelRadiusY),thisStrideY) : j-kernelRadiusY; //want modulus (positive) not the remainder
                        yEnd = std::min(prevDimensY-kernelRadiusY+j,prevDimensY); //When j>=kernelRadius, it reaches the end item. We don't care about the stride as this is the upper bound
                    }
                    else{
                        yStart = j;
                        yEnd = prevDimensY-kernelSizeY+j+1;
                    }
                    for(int k=0;k<kernelSizeX;k++){ //For each element in the kernel (k,j)
                        //Add up all the activations that it sees
                        int kernelIndex = kernelRow + k;
                        float sum = 0;
                        int thisY,thisX;
                        thisY = thisX = 0;
                        
                        int xStart, xEnd;
                        if(padding){
                            xStart = (k<kernelRadiusX)? floorMod((k-kernelRadiusX),thisStrideX) : k-kernelRadiusX;
                            xEnd = std::min(prevDimensX-kernelRadiusX+k,prevDimensX); //Same here - makes sense with a drawing
                            //The limits are needed as we have removed the padding and so we have to stop it earlier
                        }
                        else{
                            xStart = k;
                            xEnd = prevDimensX-kernelSizeX+k+1;
                        }
                        int strideOffset = xStart%thisStrideX;
                        int xHeadstart = xStart/thisStrideX;
                        float* __restrict__ stridedPrevMapOffset = stridedPrevMapChannel+strideOffset*stridedPrevMapChildSizes[1]+xHeadstart;
                        for(int y=yStart;y<yEnd;y+=thisStrideY){  //For every pixel in the previous layer (x,y) which then corresponds to one in the current (x-k,y-j)
                            int currMapRow = currMapChannel + thisY*currMapsChildSizes1;
                            int prevMapRow = prevMapChannel + y*prevMapsChildSizes1;
                            float* __restrict__ currMapDcDzsRowBase = currDcDzsData+currMapRow;
                            float* __restrict__ stridedPrevMapRow = stridedPrevMapOffset+y*stridedPrevMapChildSizes[2];
                            int x=xStart;
                            __m256 acc = _mm256_set1_ps(0);
                            for(;x+thisStrideX7<xEnd;x+=thisStrideX8){
                                float* __restrict__ currMapDcDzsBasePtr = currMapDcDzsRowBase + thisX;
                                const __m256 prevMapVals = _mm256_loadu_ps(stridedPrevMapRow+thisX);      
                                const __m256 currMapDerivs = _mm256_loadu_ps(currMapDcDzsBasePtr);
                                
                                //Add it (dC/dx*dx/dk) to kernel derivative
                                acc = _mm256_fmadd_ps(prevMapVals,currMapDerivs,acc);
        
                                thisX+=8;
                            }
                            sum += horizontalSum(acc);
                            //scalar tail
                            for(;x<xEnd;x+=thisStrideX){
                                const int currMapIndex = currMapRow + thisX;
                                const int prevMapIndex = prevMapRow + x;
                                const float reusable = currDcDzsData[currMapIndex];
                                sum += (prevMapData[prevMapIndex]) * reusable;//The previous activation
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
        #if PROFILING
            if(parentTimer){
                backwardsConvLoopTimer->stop();
            }
        #endif
    }
    
    free(currDcDzsData);
}

void CNN::finalPoolingConvBackwards(std::vector<Tensor>& dcDzs,std::vector<Tensor>& dcDxs,bool padding){
    int lastMapsL = maps.size()-1;
    int prevMapsL = maps.size()-2;
    int lastKernelsL = kernels.size()-1;
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
    int prevDimensX = mapDimens[prevMapsL].w;
    int prevDimensY = mapDimens[prevMapsL].h;
    int currDimensX = mapDimens[lastMapsL].w;
    int currDimensY = mapDimens[lastMapsL].h;
    int kernelSizeX = kernelSizes[lastKernelsL].second;
    int kernelSizeY = kernelSizes[lastKernelsL].first;
    int kernelRadiusX = (int) floor(kernelSizeX/2);
    int kernelRadiusY = (int) floor(kernelSizeY/2);
    int thisStrideX = strides[strides.size()-2].second;
    int thisStrideY = strides[strides.size()-2].first;
    int poolWidth = mapDimens[lastMapsL].w/strides[strides.size()-1].second;
    int poolHeight = mapDimens[lastMapsL].h/strides[strides.size()-1].first;
    int poolArea = poolWidth*poolHeight;
    std::vector<int> lastMapsChildSizes = maps[lastMapsL].getChildSizes();
    std::vector<int> prevMapsChildSizes = maps[prevMapsL].getChildSizes();
    std::vector<int> lastKernelsChildSizes = kernels[lastKernelsL].getChildSizes();
    const int prevMapsChildSizes1 = prevMapsChildSizes[1];
    //don't count the max pixel more than once
    //ChatGPT says uint8_t is quicker than bool as bool does bit packing
    for(int i=0;i<mapDimens[lastMapsL].c;i++){ //for each final map
        const int mlpRegion = i*poolArea; 
        int kernelToChannel = i*lastKernelsChildSizes[0];
        for(int prevMapI=0;prevMapI<mapDimens[prevMapsL].c;prevMapI++){
            int prevMapChannel = prevMapI*prevMapsChildSizes[0];
            int kernelFromChannel = kernelToChannel + prevMapI*lastKernelsChildSizes[1];
            for(int j=0;j<kernelSizeY;j++){
                int kernelRow = kernelFromChannel + j*lastKernelsChildSizes[2];
                int yStart, yEnd;
                if(padding){
                    yStart = (j<kernelRadiusY)? floorMod((j-kernelRadiusY),thisStrideY) : j-kernelRadiusY; //want modulus (positive) not the remainder
                    yEnd = std::min(prevDimensY-kernelRadiusY+j,prevDimensY); //When j>=kernelRadius, it reaches the end item. We don't care about the stride as this is the upper bound
                }
                else{
                    yStart = j;
                    yEnd = prevDimensY-kernelSizeY+j+1;
                }

                for(int k=0;k<kernelSizeX;k++){ //For each element in the kernel (k,j)
                    int kernelIndex = kernelRow + k;
                    //Add up all the activations that it sees
                    float sum = 0;
                    int xStart, xEnd;
                    if(padding){
                        xStart = (k<kernelRadiusX)? floorMod((k-kernelRadiusX),thisStrideX) : k-kernelRadiusX;
                        xEnd = std::min(prevDimensX-kernelRadiusX+k,prevDimensX); //Same here - makes sense with a drawing
                        //The limits are needed as we have removed the padding and so we have to stop it earlier
                    }
                    else{
                        xStart = k;
                        xEnd = prevDimensX - kernelSizeX+k+1;
                    }
                    for(int r=0;r<poolArea;r++){
                        int mlpIndex = mlpRegion + r;
                        //If this activation has been dropped out
                        if(dcDzs0Data[mlpIndex]==0.0f) continue;
                        int maxPixelIndex = maxPoolIndicesData[mlpIndex];
                        //Curr layer indices
                        int thisY = maxPixelIndex/currDimensY;
                        int thisX = maxPixelIndex - thisY*currDimensX;
                        //Prev layer indices
                        int y = yStart + thisY*thisStrideY;
                        int x = xStart + thisX*thisStrideX;
                        if (y >= yEnd || x >= xEnd) {
                            //If this kernel element doesn't touch a real pixel
                            //Occurs when we've padded and so x and y are out of bounds (in the padding)
                            //We set xStart and yStart such that it can't happen at the start
                            continue;
                        }      
                        int prevMapIndex = prevMapChannel + y*prevMapsChildSizes1 + x;
                        //In the first MLP layer a=relu(x) where x is the max activation pixel from pooling
                        sum += prevMapsData[prevMapIndex] * dcDzs0Data[mlpIndex]; //The activation of the previous layer * the correct derivative from pooling
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
    int prevDimensX = mapDimens[lSub1].w;
    int prevDimensY = mapDimens[lSub1].h;
    int currDimensX = mapDimens[l].w;
    int currDimensY = mapDimens[l].h;
    int kernelSizeX = kernelSizes[lSub1].second;
    int kernelSizeY = kernelSizes[lSub1].first;
    int kernelRadiusX = (int) floor(kernelSizeX/2);
    int kernelRadiusY = (int) floor(kernelSizeY/2);
    int poolStrideX = strides[l].second;
    int poolStrideY = strides[l].first;
    int thisStrideX = strides[lSub1].second;
    int thisStrideY = strides[lSub1].first;
    int poolDimensX = mapDimens[lPlus1].w;
    int poolDimensY = mapDimens[lPlus1].h;
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
    for(int i=0;i<mapDimens[l].c;i++){ //prev (l-1) --conv-> curr (l) --pool-> pooled (l+1)
        //pooling is 1:1 between channels
        int currMapChannel = i*currMapsChildSizes[0];
        int pooledMapChannel = i*pooledMapsChildSizes[0];
        int kernelToChannel = i*kernelsChildSizes[0];
        for(int prevMapI=0;prevMapI<mapDimens[l-1].c;prevMapI++){
            int kernelFromChannel = kernelToChannel + prevMapI*kernelsChildSizes[1];
            int prevMapChannel = prevMapI*prevMapsChildSizes[0];
            for(int j=0;j<kernelSizeY;j++){
                int kernelRow = kernelFromChannel + j*kernelsChildSizes[2];
                int yStart, yEnd;
                if(padding){
                    yStart = (j<kernelRadiusY)? floorMod((j-kernelRadiusY),thisStrideY) : j-kernelRadiusY; //want modulus (positive) not the remainder
                    yEnd = std::min(prevDimensY-kernelRadiusY+j,prevDimensY); //When j>=kernelRadius, it reaches the end item. We don't care about the stride as this is the upper bound
                }
                else{
                    yStart = j;
                    yEnd = prevDimensY-kernelSizeY+j+1;
                }
                for(int k=0;k<kernelSizeX;k++){ //For each element in the kernel (k,j)
                    int kernelIndex = kernelRow + k;
                    //Add up all the activations that it sees
                    float sum = 0;
                    int thisY,thisX;
                    thisY = thisX = 0;
                    std::vector<std::vector<bool>> done(poolDimensY,std::vector<bool>(poolDimensX));
                    int xStart, xEnd;
                    if(padding){
                        xStart = (k<kernelRadiusX)? floorMod((k-kernelRadiusX),thisStrideX) : k-kernelRadiusX;
                        xEnd = std::min(prevDimensX-kernelRadiusX+k,prevDimensX); //Same here - makes sense with a drawing
                        //The limits are needed as we have removed the padding and so we have to stop it earlier
                    }
                    else{
                        xStart = k;
                        xEnd = prevDimensX - kernelSizeX+k+1;
                    }
                    for(int y=yStart;y<yEnd;y+=thisStrideY){  //For every pixel in the previous layer (x,y) which then corresponds to one in the current (x-k,y-j)
                        int poolY = ((thisY)/poolStrideY);
                        int currMapRow = currMapChannel + thisY*currMapsChildSizes[1];
                        int pooledMapRow = pooledMapChannel + poolY*pooledMapsChildSizes[1];
                        int prevMapRow = prevMapChannel + y*prevMapsChildSizes[1];
                        for(int x=xStart;x<xEnd;x+=thisStrideX){ //Derivatve of the corresponding pixel in the next (backwards) layer
                            int poolX = ((thisX)/poolStrideX);
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
        std::vector<std::vector<bool>> done(poolDimensY,std::vector<bool>(poolDimensX));
        for(int y=0;y<currDimensY;y++){
            int poolY = (y/poolStrideY);
            int currMapRow = currMapChannel + y*currMapsChildSizes[1];
            int pooledMapRow = pooledMapChannel + poolY*pooledMapsChildSizes[1];
            for(int x=0;x<currDimensX;x++){
                int poolX = (x/poolStrideX);
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




