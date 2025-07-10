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
    verbose = false;

    this->d = dataset;
    this->kernels = loadKernels(restart);
    this->weights = loadWeights(restart);
    kernelsGrad = arrayCopy5D(kernels,0);
    weightsGrad = arrayCopy3D(weights,0);
    this->LR = LR;
    this->activations = newMatrix<d1>({numNeurons.size(),0});
    for(int l=0;l<numNeurons.size();l++){
        activations[l] = vector<int>(numNeurons[l]);
    }
    this.maps = new float[numMaps.size()][][][];
    for(int l=0;l<numMaps.size();l++){
        maps[l] = new float[numMaps[l]][mapDimens[l]][mapDimens[l]];
    }
    if(restart){
        resetKernels();
        resetWeights();
    }
}

//Creating a copy from a template CNN
CNN::CNN(CNN template,float LR,Dataset dataset) {
    numNeurons = template.numNeurons;
    numMaps = template.numMaps;
    mapDimens = template.mapDimens;
    kernelSizes = template.kernelSizes;
    strides = template.strides;
    padding = template.padding;
    verbose = template.verbose;
    this.d = dataset;
    this.kernels = arrayCopy5D(template.kernels);
    this.weights = arrayCopy3D(template.weights);
    kernelsGrad = arrayCopy5D(kernels,0);
    weightsGrad = arrayCopy3D(weights,0);
    this.LR = LR;
    this.activations = new float[numNeurons.size()][];
    for(int l=0;l<numNeurons.size();l++){
        activations[l] = new float[numNeurons[l]];
    }
    this.maps = new float[numMaps.size()][][][];
    for(int l=0;l<numMaps.size();l++){
        maps[l] = new float[numMaps[l]][mapDimens[l]][mapDimens[l]]; //Already initialised to 0
    }
}


//----------------------------------------------------
//KEY METHODS 


std::string CNN::forwards(int[][][] imageInt){
    long startTime = System.currentTimeMillis();
    reset();
    maps[0] = parseImg(imageInt);
    maps[0] = normaliseImg(maps[0],d.getPixelMeans(),d.getPixelStdDevs());
    //Convolutional and pooling layers
    for(int l=1;l<numMaps.size();l++){
        for(int i=0;i<numMaps[l];i++){
            if(kernelSizes[l-1]==0){
                maps[l][i] = maxPool(maps[l-1][i],strides[l-1],strides[l-1]); //maxPool requires 1:1 channels between layers
            }
            else{   
                maps[l][i] = convolution(maps[l-1], kernels[l-1][i],strides[l-1],strides[l-1],padding);
            }
            
        }
    }
    //Final pooling 
    float[][][] pooled = new float[numMaps[numMaps.size()-1]][][];
    for(int i=0;i<numMaps[numMaps.size()-1];i++){
        pooled[i] = maxPool(maps[numMaps.size()-1][i],strides[strides.size()-1],strides[strides.size()-1]);
        for(int y=0;y<pooled[i].size();y++){
            System.arraycopy(pooled[i][y], 0, activations[0], i*pooled[i].size()*pooled[i][y].size()+y*pooled[i][y].size(), pooled[i][y].size());
            //(I nicked this from the "Quick fix" - it's probably quicker than my implementation and it stops shouting at me)
        }
    }
    //MLP
    for(int l=0;l<weights.size();l++){
        for(int i=0;i<numNeurons[l+1];i++){
            for(int j=0;j<numNeurons[l];j++){
                activations[l+1][i] += activations[l][j] * weights[l][i][j]; 
            }
            activations[l+1][i] = activations[l+1][i] + weights[l][i][activations[l].size()]; //add bias
            if(l!=weights.size()-1){ //We'll softmax the last layer and so relu is unnecessary
                activations[l+1][i] = leakyRelu(activations[l+1][i]);
            }
        }
    }
    activations[activations.size()-1] = softmax(activations[activations.size()-1]);
    float largestActivation = activations[activations.size()-1][0];
    int result = 0;
    for(int i=1;i<numNeurons[numNeurons.size()-1];i++){
        if(activations[activations.size()-1][i]>largestActivation){
            largestActivation = activations[activations.size()-1][i];
            result = i;
        }
    }
    if(verbose){
        System.out.println(Arrays.toString(activations[activations.size()-1]));
        System.out.println("Forwards took "+(System.currentTimeMillis()-startTime)+"ms");
    }
    return d.plantNames.get(result);
}

void CNN::backwards(int[][][] imageInt,String answer){ //adds the gradient to its internal gradient arrays
    forwards(imageInt); //set all the activations
    //Gradients are not reset each time to enable batches
    long mlpStart = System.currentTimeMillis();
    //MLP derivs
    Integer correctOutput = d.plantToIndex.get(answer);
    if(correctOutput==null){
        System.out.println("\""+answer+"\" does not exist");
        return; 
    }
    float[][] dcDzs = new float[numNeurons.size()][]; //z is the pre-activation summations 
    //The derivative includes the activation derivative
    //z_i = w_j_i*a_j + ... + b_i
    for(int l=0;l<numNeurons.size();l++){
        dcDzs[l] = new float[numNeurons[l]]; //all layers need activation derivatives
    }
    int lastLayer = numNeurons.size()-1;
    for(int i=0;i<numNeurons[lastLayer];i++){
        if(Float.isNaN(activations[lastLayer][i])){
            System.out.println("Invalid activation in last layer at i:"+i);
            return;
        }
        //Cross entropy loss
        dcDzs[lastLayer][i] = activations[lastLayer][i] - ((i==correctOutput)?1:0); 
    }

    mlpBackwards(dcDzs); 
    if(verbose) System.out.println("Backwards MLP took "+String.valueOf(System.currentTimeMillis()-mlpStart)+"ms");
    //x is the image pixel value and so these dcDxs are the derivatives based on pixels which are carried backwards
    float[][][][] dcDxs = new float[numMaps.size()-2][][][];
    //No dcDxs for first or last (last goes straight into the MLP)
    for(int l=0;l<numMaps.size()-2;l++){
        if(kernelSizes[l+1]==0){
            //There doesn't need to be any dcDxs for any pre-pooling maps
            //Blank dcDxs for pre-pooling layers so that indices stay consistent with map indices
            dcDxs[l] = new float[0][0][0];
        }
        else{
            dcDxs[l] = new float[numMaps[l+1]][mapDimens[l+1]][mapDimens[l+1]];
        }
        
    }
    long convolutionStart = System.currentTimeMillis();
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
    if(verbose){
        System.out.println("Backwards Convolution took "+String.valueOf(System.currentTimeMillis()-convolutionStart)+"ms");
        System.out.println("Backwards took "+String.valueOf(System.currentTimeMillis()-mlpStart)+"ms");
    }
}


//----------------------------------------------------
//BACKPROPAGATION-RELATED

void CNN::mlpBackwards(float[][] dcDzs){
    for(int l=weights.size()-1;l>=0;l--){
        for(int i=0;i<numNeurons[l+1];i++){
            for(int j=0;j<numNeurons[l];j++){//NOTE: Weights gradient != negative gradient
                weightsGrad[l][i][j] += dcDzs[l+1][i] * activations[l][j];
                //dC/dw = dC/da_i+1 * da_i+1/dz * dz/dw
                dcDzs[l][j] +=  dcDzs[l+1][i] * weights[l][i][j] * ((activations[l][j]<=0)?0.01f:1);//next layer
                //dC/dz_i = dC/dz_i+1 * dz_i+1/da_i * da_i/dz_i
            }
            //bias
            weightsGrad[l][i][activations[l].size()] += dcDzs[l+1][i];
        }
    }
}

void CNN::convBackwards(float[][][][] dcDxs, int l,boolean padding){
    int lSub1 = l-1;
    int lSub2 = l-2;
    int prevDimens = mapDimens[lSub1];
    int currDimens = mapDimens[l];
    int kernelSize = kernelSizes[lSub1];
    int kernelRadius = (int) Math.floor(kernelSize/2);
    int thisStride = strides[lSub1];
    for(int i=0;i<numMaps[l];i++){ //For each convolution output
        for(int prevMapI=0;prevMapI<numMaps[lSub1];prevMapI++){ //For each previous channel
            for(int j=0;j<kernelSize;j++){
                for(int k=0;k<kernelSize;k++){ //For each element in the kernel (k,j)
                    //Add up all the activations that it sees
                    float sum = 0f;
                    float reusable;
                    int thisY,thisX;
                    thisY = thisX = 0;
                    int yStart, yEnd;
                    int xStart, xEnd;
                    if(padding){
                        yStart = (j<kernelRadius)? Math.floorMod((j-kernelRadius),thisStride) : j-kernelRadius; //want modulus (positive) not the remainder
                        yEnd = Math.min(prevDimens-kernelRadius+j,prevDimens); //When j>=kernelRadius, it reaches the end item. We don't care about the stride as this is the upper bound
                        xStart = (k<kernelRadius)? Math.floorMod((k-kernelRadius),thisStride) : k-kernelRadius;
                        xEnd = Math.min(prevDimens-kernelRadius+k,prevDimens); //Same here - makes sense with a drawing
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
                            if(!floatCmp(dcDxs[lSub1][i][thisY][thisX],0f)){ //yes, l-1 is correct (dcDxs only has numMaps.size()-2 layers)
                                reusable =   dcDxs[lSub1][i][thisY][thisX] //Previous derivative (from pooling)
                            *  ((maps[l][i][thisY][thisX])<=0f?0.01f:1); //*Leaky Relu Derivative
                                sum += (maps[lSub1][prevMapI][y][x]) *reusable;//The previous activation
                                if(l!=1) dcDxs[lSub2][prevMapI][y][x] += reusable * kernels[lSub1][i][prevMapI][j][k]; //don't have dcDxs for the first layer
                            }
                            thisX++;
                        }
                        thisX = 0;
                        thisY++;
                    }
                    kernelsGrad[lSub1][i][prevMapI][j][k] += sum; //kernels are [layer][nextLayerChannel][prevLayerChannel]
                }
            }
        }
        //Bias doesn't care about the inputs and so only needs the output
        float biasSum = 0f;
        for(int y=0;y<currDimens;y++){
            for(int x=0;x<currDimens;x++){
                //Bias has to be here as otherwise it would count the same pixels multiple times
                biasSum += dcDxs[lSub1][i][y][x] * ((maps[l][i][y][x])<=0?0.01f:1); //Bias deriv = cost deriv * relu deriv * 1 (only 1 bias term in each new pixel expression
            }            
        }
        kernelsGrad[lSub1][i][numMaps[lSub1]-1][kernelSize-1][kernelSize] += biasSum;
    }
}

void CNN::finalPoolingConvBackwards(float[][] dcDzs,float[][][][] dcDxs,boolean padding){
    for(int i=0;i<numMaps[numMaps.size()-1];i++){ //for each final map
        int prevDimens = mapDimens[numMaps.size()-2];
        int currDimens = mapDimens[numMaps.size()-1];
        int kernelSize = kernelSizes[kernelSizes.size()-1];
        int kernelRadius = (int) Math.floor(kernelSize/2);
        int poolStride = strides[strides.size()-1];
        int thisStride = strides[strides.size()-2];
        int poolWidth = mapDimens[numMaps.size()-1]/strides[strides.size()-1];
        int poolArea = poolWidth*poolWidth;
        int mlpRegion = i*poolArea; //Saving a few multiplications
        for(int prevMapI=0;prevMapI<numMaps[numMaps.size()-2];prevMapI++){
            for(int j=0;j<kernelSize;j++){
                for(int k=0;k<kernelSize;k++){ //For each element in the kernel (k,j)
                    //Add up all the activations that it sees
                    float sum = 0f;
                    int thisY,thisX,mlpIndex,mlpSubIndex;
                    thisY = thisX = 0;
                    boolean[] done = new boolean[poolArea]; //don't count the max pixel more than once
                    int yStart, yEnd;
                    int xStart, xEnd;
                    if(padding){
                        yStart = (j<kernelRadius)? Math.floorMod((j-kernelRadius),thisStride) : j-kernelRadius; //want modulus (positive) not the remainder
                        yEnd = Math.min(prevDimens-kernelRadius+j,prevDimens); //When j>=kernelRadius, it reaches the end item. We don't care about the stride as this is the upper bound
                        xStart = (k<kernelRadius)? Math.floorMod((k-kernelRadius),thisStride) : k-kernelRadius;
                        xEnd = Math.min(prevDimens-kernelRadius+k,prevDimens); //Same here - makes sense with a drawing
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
                            if(floatCmp(maps[maps.size()-1][i][thisY][thisX],activations[0][mlpIndex]) && !done[mlpSubIndex]){ //only the max element has a derivative
                                done[mlpSubIndex] = true;
                                //In the first MLP layer a=relu(x) where x is the max activation pixel from pooling
                                dcDxs[dcDxs.size()-1][prevMapI][y][x] += dcDzs[0][mlpIndex] * kernels[kernels.size()-1][i][prevMapI][j][k] ;//*kernel weight
                                sum+=maps[maps.size()-2][prevMapI][y][x]*dcDzs[0][mlpIndex]; //The activation of the previous layer * the correct derivative from pooling
                            }
                            thisX++;
                        }
                        thisX = 0;
                        thisY++;
                    }
                    kernelsGrad[kernels.size()-1][i][prevMapI][j][k] += sum;
                }
            }
        }
        float biasSum = 0f;
        int mlpIndex,mlpSubIndex;
        boolean[] done = new boolean[poolArea];
        for(int y=0;y<currDimens;y++){
            for(int x=0;x<currDimens;x++){
                mlpSubIndex = (((y)/poolStride)*poolWidth) + ((x)/poolStride);
                mlpIndex = mlpRegion + mlpSubIndex;
                //Bias has to be here as otherwise it would count the same pixels multiple times
                if(floatCmp(maps[maps.size()-1][i][y][x],activations[0][mlpIndex]) && !done[mlpSubIndex]){
                    done[mlpSubIndex] = true;
                    biasSum += dcDzs[0][mlpIndex]; //Bias deriv = cost deriv * relu deriv * 1 (only 1 bias term in each new pixel expression
                }
            }
        }
        kernelsGrad[kernels.size()-1][i][numMaps[numMaps.size()-2]-1][kernelSize-1][kernelSize] += biasSum;
    }
}

void CNN::poolingConvBackwards(float[][][][] dcDxs, int l,boolean padding){
    for(int i=0;i<numMaps[l];i++){ //prev (l-1) --conv-> curr (l) --pool-> pooled (l+1)
        //pooling is 1:1 between channels
        int prevDimens = mapDimens[l-1];
        int currDimens = mapDimens[l];
        int kernelSize = kernelSizes[l-1];
        int kernelRadius = (int) Math.floor(kernelSize/2);
        int poolStride = strides[l];
        int thisStride = strides[l-1];
        int poolDimens = mapDimens[l+1];
        for(int prevMapI=0;prevMapI<numMaps[l-1];prevMapI++){
            for(int j=0;j<kernelSize;j++){
                for(int k=0;k<kernelSize;k++){ //For each element in the kernel (k,j)
                    //Add up all the activations that it sees
                    float sum = 0f;
                    float reusable;
                    int thisY,thisX,poolX,poolY;
                    thisY = thisX = 0;
                    boolean[][] done = new boolean[poolDimens][poolDimens];
                    int yStart, yEnd;
                    int xStart, xEnd;
                    if(padding){
                        yStart = (j<kernelRadius)? Math.floorMod((j-kernelRadius),thisStride) : j-kernelRadius; //want modulus (positive) not the remainder
                        yEnd = Math.min(prevDimens-kernelRadius+j,prevDimens); //When j>=kernelRadius, it reaches the end item. We don't care about the stride as this is the upper bound
                        xStart = (k<kernelRadius)? Math.floorMod((k-kernelRadius),thisStride) : k-kernelRadius;
                        xEnd = Math.min(prevDimens-kernelRadius+k,prevDimens); //Same here - makes sense with a drawing
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
                            if(floatCmp(maps[l][i][thisY][thisX],maps[l+1][i][poolY][poolX]) && !done[poolY][poolX] && !floatCmp(dcDxs[l][i][poolY][poolX],0f)){ //only the max element has a derivative
                                done[poolY][poolX] = true;
                                reusable =  dcDxs[l][i][poolY][poolX]//Previous derivative (from pooling)
                                * ((maps[l][i][thisY][thisX])<=0?0.01f:1); //*Leaky Relu Derivative
                                if(l!=1) dcDxs[l-2][i][y][x] += reusable * kernels[l-1][i][prevMapI][j][k] ;//*kernel weight
                                sum+=maps[l-1][prevMapI][y][x]*reusable; //The activation of the previous layer * the correct derivative from pooling
                            }
                            thisX++;
                        }
                        thisX = 0;
                        thisY++;
                    }
                    kernelsGrad[l-1][i][prevMapI][j][k] += sum;
                }
            }
        }
        float biasSum = 0f;
        int poolX,poolY;
        boolean[][] done = new boolean[poolDimens][poolDimens];
        for(int y=0;y<currDimens;y++){
            for(int x=0;x<currDimens;x++){
                poolY = (y/poolStride);
                poolX = (x/poolStride);
                //Bias has to be here as otherwise it would count the same pixels multiple times
                if(floatCmp(maps[l][i][y][x],maps[l+1][i][poolY][poolX]) && !done[poolY][poolX]){
                    done[poolY][poolX] = true;
                    biasSum += dcDxs[l][i][poolY][poolX] * ((maps[l][i][y][x])<=0?0.01f:1); //Bias deriv = cost deriv * relu deriv * 1 (only 1 bias term in each new pixel expression)
                }
            }
        }
        kernelsGrad[l-1][i][numMaps[l-1]-1][kernelSize-1][kernelSize] += biasSum;
    }
}




