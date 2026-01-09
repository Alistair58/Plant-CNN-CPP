#include <string>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include "cnn.hpp"
#include "cnnutils.hpp"
#include "dataset.hpp"
#include "plantimage.hpp"
#include "globals.hpp"
#include "utils.hpp"
#include "tensor.hpp"
#include <numbers>

//Default values
static float LR = 0.02;
static int batchSize = 64;
#define TRAIN 1
#define TEST 2

#if PROFILING
    #include "timer.hpp"
#endif



std::atomic<int> missedCount{0};

static void trainBatch(CNN *n, Dataset *d, int batchSize,int numImageThreads,std::vector<CNN*>& cnns
#if PROFILING
    ,Timer *parentTimer = nullptr
#endif
);

static void train(CNN *n, Dataset *d, int numBatches,int batchSize,int numImageThreads, int numCnnThreads);
static void test(CNN *n, Dataset *d, int numTest,bool testOnTrainingData,bool outputIndividuals);

//DONE


//TODO
//Fix large gradient problem
// - Run it again
// - More unit tests
//Test new larger model
//Speed up



int main(int argc,char **argv){
    //("train"|"test") 
    //"train" ->       {numBatches} (rs=(true|false))? (bs={batchSize})? (lr={LR})?
    //"test"  ->       {numTestImages} (lr={LR})? (ds=(test|train))? (out=(true|false))?
    //out => output every single test result
    Dataset *d = new Dataset(datasetDirPath,0.8f);
    CNN *cnn = nullptr;
    const int numImageThreads = 6;
    const int numCnnThreads = 16;
    int mode = -1;
    bool restart = false;
    //TODO turn back on dropout
    float dropoutProbability = 0;

    /*
    Model 5:
        128x128x3
        64x64x30 (3x3 conv stride 2)
        32x32x60 (3x3 conv stride 2)
        16x16x120 (3x3 conv stride 2)
        4x4x120 (max pool)
        1920
        1920 (FC)
        47 (FC)
    */
    std::vector<int> numNeurons = {1920,1920,47};
    //includes the result of pooling (except final pooling)
    std::vector<dimens> mapDimens(4);
    mapDimens[0] = {3,128,128};
    mapDimens[1] = {30,64,64};
    mapDimens[2] = {60,32,32};
    mapDimens[3] = {120,16,16};
    //0 represents a pooling layer, the last one is excluded
    std::vector<std::pair<int,int>> kernelSizes(3);
    kernelSizes[0] = {3,3}; //h,w
    kernelSizes[1] = {3,3};
    kernelSizes[2] = {3,3};
    //pooling strides are included
    std::vector<std::pair<int,int>> strides(4);
    strides[0] = {2,2};//y,x - pooling strides are included
    strides[1] = {2,2};
    strides[2] = {2,2};
    strides[3] = {4,4};

    bool padding = true;


    if(argc<3){
        throw std::invalid_argument("argv must contain at least 2 arguments");
    }
    if(toLower(argv[1]) == toLower("train")) mode = TRAIN;
    else if(toLower(argv[1]) == toLower("test")) mode = TEST;
    else{
        throw std::invalid_argument("Argument 1 must either be \"train\" or \"test\"");
    }
    if(mode==TRAIN){
        int numBatches = atoi(argv[2]);
        if(numBatches<=0){
            throw std::invalid_argument("For train, argument 2 must be the number of batches (a positive integer)");
        }
        for(int i=3;i<argc;i++){
            std::vector<std::string> splitRes = strSplit(argv[i],{'='});
            if(splitRes.size()!=2){
                throw std::invalid_argument("Optional arguments must be in the format {parameter}={value}");
            }
            if(toLower(splitRes[0])=="rs"){
                if(toLower(splitRes[1])=="true"){
                    std::cout << ANSI_RED << "Are you sure you want to reset the model to random weights? (Y/N) " << ANSI_RESET << std::endl;
                    while(true){
                        std::string userRes;
                        std::cin >> userRes;
                        if(toLower(userRes)=="y"){
                            restart = true;
                            std::cout << "Model reset" << std::endl;
                            break;
                        }
                        else if(toLower(userRes)=="n"){
                            restart = false;
                            std::cout << "Model reset aborted" << std::endl;
                            break;
                        }
                        else{
                            std::cout << "Enter Y or N" << std::endl;
                        }
                    }
                }
                else if(toLower(splitRes[1])=="false"){
                    restart = false;
                }   
                else{
                    throw std::invalid_argument("Parameter \"rs\" (restart) can only be set to \"true\" or \"false\"");
                }
            }
            else if(toLower(splitRes[0])=="bs"){
                batchSize = stoi(splitRes[1]);
                if(batchSize<=0){
                    throw std::invalid_argument("Parameter \"bs\" (batch size) must be a positive integer");
                }
            }
            else if(toLower(splitRes[0])=="lr"){
                LR = stof(splitRes[1]);
                if(LR<=0.0f){
                    throw std::invalid_argument("Parameter \"lr\" (learning rate) must be a positive float");
                }
            }
            else{
                throw std::invalid_argument("Optional argument "+std::to_string(i)+"'s parameter \""+splitRes[0]+"\" is invalid for train");
            }
        }
        cnn = new CNN(
            LR,d,restart,dropoutProbability,
            numNeurons,mapDimens,kernelSizes,strides,padding
        );
        train(cnn,d,numBatches,batchSize,numImageThreads,numCnnThreads);
    }
    if(mode==TEST){
        int numTestImages = atoi(argv[2]);
        int testOnTrainingData = false;
        bool outputIndividuals = false;
        if(numTestImages<=0){
            throw std::invalid_argument("For test, argument 2 must be the number of test images (a positive integer)");
        }
        for(int i=3;i<argc;i++){
            std::vector<std::string> splitRes = strSplit(argv[i],{'='});
            if(splitRes.size()!=2){
                throw std::invalid_argument("Optional arguments must be in the format {parameter}={value}");
            }
            if(toLower(splitRes[0])=="lr"){
                LR = stof(splitRes[1]);
                if(LR<=0.0f){
                    throw std::invalid_argument("Parameter \"lr\" (learning rate) must be a positive float");
                }
            }
            else if(toLower(splitRes[0])=="ds"){
                if(toLower(splitRes[1])=="train"){
                    testOnTrainingData = true;
                }
                else if(toLower(splitRes[1])=="test"){
                    testOnTrainingData = false;
                }   
                else{
                    throw std::invalid_argument("Parameter \"ds\" (dataset) can only be set to \"train\" or \"test\"");
                }
            }
            else if(toLower(splitRes[0])=="out"){
                if(toLower(splitRes[1])=="true"){
                    outputIndividuals = true;
                }
                else if(toLower(splitRes[1])=="false"){
                    outputIndividuals = false;
                }   
                else{
                    throw std::invalid_argument("Parameter \"out\" (output individuals) can only be set to \"true\" or \"false\"");
                }
            }
            else{
                throw std::invalid_argument("Optional parameter \""+splitRes[0]+"\" is invalid for test");
            }
        }
        cnn = new CNN(
            LR,d,false,0.0,
            numNeurons,mapDimens,kernelSizes,strides,padding
        );
        test(cnn,d,numTestImages,testOnTrainingData,outputIndividuals);
    }
    delete d;
    delete cnn;
}
    
static void test(CNN *n, Dataset *d, int numTest,bool testOnTrainingData,bool outputIndividuals){
    #if PROFILING
        Timer testTimer = Timer("test");
    #endif
    int correctCount = 0;
    for (int i=0;i<numTest;i++) {
        //true for test data and false for training data
        PlantImage pI = d->randomImageObj(!testOnTrainingData);
        if(pI.label.length()!=0){
            std::string response = n->forwards(pI.data,false
            #if PROFILING
                ,&testTimer
            #endif
            );
            bool correct = response==pI.label;
            if(outputIndividuals){
                std::cout << ("("+ std::to_string(i+1) + "/" + std::to_string(numTest)+")  "+((correct)?ANSI_GREEN:ANSI_RED)+
                pI.label +" ("+std::to_string(pI.index)+ ") Computer said: " + response+ANSI_RESET) << std::endl;
            }
            else{
                int percentageComplete = (float)i/numTest*100;
                printf("\r[%.*s%.*s] (%d/%d) %d%% complete",
                    percentageComplete/10, "##########",
                    10-percentageComplete/10, "          ",
                    i,numTest,
                    percentageComplete
                );
                //stdout is line-buffered but we're not writing a new line with \n and so flush
                fflush(stdout);
            }
            if(correct) correctCount++;
        }
        else i--;
    }
    std::cout << ("\r"+ANSI_CLEAR_LINE+"Accuracy: "+std::to_string((float)correctCount*100/numTest)+"%") <<std::endl;
    #if PROFILING
        testTimer.stop();
        testTimer.output();
    #endif
}

static void train(CNN *n, Dataset *d, int numBatches,int batchSize,int numImageThreads, int numCnnThreads){
    #if PROFILING
        Timer trainTimer = Timer("train");
    #endif
    uint64_t startTime = getCurrTimeMs();
    std::vector<CNN*> cnns(numCnnThreads);
    cnns[0] = n;
    int savePeriod = 25;
    for(int i=1;i<numCnnThreads;i++){
        //shallow copy of weights and kernels
        //Must be shallow as apply gradients only updates cnn[0]'s weights and kernels
        cnns[i] = new CNN(n,LR,d,false); 
    }
    for(int i=0;i<numBatches;i++){ // numBatches of batchSize
        trainBatch(n, d, batchSize,numImageThreads,cnns
        #if PROFILING
            ,&trainTimer
        #endif
        );
        if(i%savePeriod == 0 && i>0){ //save every 25 batches
            n->saveKernels(
            #if PROFILING
                &trainTimer
            #endif
            );
            n->saveWeights(
            #if PROFILING
                &trainTimer
            #endif
            );
        }
        int percentageComplete = (float)i/numBatches*100;
        std::string lastSavedStr = "  Most recent save at batch " + std::to_string((i/savePeriod)*savePeriod);
        printf("\r[%.*s%.*s] (%d/%d) %d%% complete %.*s",
            percentageComplete/10, "##########",
            10-percentageComplete/10, "          ",
            i,numBatches,
            percentageComplete,
            i>=savePeriod?(int)lastSavedStr.length():0,lastSavedStr.c_str()
        );
        //stdout is line-buffered but we're not writing a new line with \n and so flush
        fflush(stdout);
    }
    n->saveWeights(
    #if PROFILING
        &trainTimer
    #endif
    );
    n->saveKernels(
    #if PROFILING
        &trainTimer
    #endif    
    );
    std::cout << "\r"+ANSI_CLEAR_LINE+"Done" << std::endl;
    uint64_t endTime = getCurrTimeMs();
    int secs = (int)((endTime-startTime)/1000);
    int mins = (int) (secs/60);
    int hours = (int) (mins/60);
    std::cout << "Took: "+
        std::to_string(hours)+" hr(s) "+
        std::to_string(mins%60)+" min(s) "+
        std::to_string(secs%60)+" sec(s)"
    << std::endl;
    std::cout << "Missed: "+std::to_string(missedCount.load(std::memory_order_acquire)) << std::endl;
    //start at 1 as we don't want to delete the original CNN (at index 0)
    for(int i=1;i<cnns.size();i++){
        delete cnns[i];
    }
    #if PROFILING
        trainTimer.stop();
        trainTimer.output();
    #endif
}


static void trainBatch(CNN *n, Dataset *d, int batchSize,int numImageThreads,std::vector<CNN*>& cnns
#if PROFILING
    ,Timer *parentTimer
#endif
){ 
    #if PROFILING
        Timer *batchTimer = nullptr;
        if(parentTimer) batchTimer = parentTimer->addChildTimer("batch");
    #endif


    //Single thread, sequential code for debugging
    // for(int i{};i<batchSize;i++){
    //     PlantImage *p = d->randomImage(false);
    //     if(p!=nullptr && p->index!=-1 && p->label.length()>0){
    //         n->backwards(p->data,p->label);
    //     }
    //     else missedCount.fetch_add(1, std::memory_order_relaxed);
    //     if(p!=nullptr){
    //         delete p;
    //     }
    // }
    // n->applyGradients(batchSize);

    int numCnnThreads = cnns.size();
    std::vector<std::thread> cnnThreads(numCnnThreads);
    std::vector<std::thread> imageThreads(numImageThreads);
    std::vector<std::atomic<PlantImage*>> plantImages(batchSize);
    for(int i=0;i<batchSize;i++) plantImages[i].store(nullptr,std::memory_order_relaxed);
    for(int iT=0;iT<numImageThreads;iT++){
        imageThreads[iT] = std::thread(
            [](int threadId,int batchSize,int numImageThreads,std::vector<std::atomic<PlantImage*>> *plantImages,Dataset *d
            #if PROFILING
                ,Timer *imageThreadTimer
            #endif
            ){
                for(int i=threadId;i<batchSize;i+=numImageThreads){
                    PlantImage *p = d->randomImage(false
                    #if PROFILING 
                        ,imageThreadTimer?imageThreadTimer:nullptr
                    #endif
                    );
                    (*plantImages)[i].store(p,std::memory_order_release); 
                }
                #if PROFILING 
                    if(imageThreadTimer) imageThreadTimer->stop();
                #endif
            },iT,batchSize,numImageThreads,&plantImages,d
            #if PROFILING
                ,parentTimer?batchTimer->addChildTimer("imageThread"+std::to_string(iT)):nullptr
            #endif
        );
    }
    for(int cT=0;cT<numCnnThreads;cT++){ 
        cnnThreads[cT]= std::thread(
            [](int threadId,int batchSize,int numCnnThreads,std::vector<std::atomic<PlantImage*>> *plantImages,Dataset *d,std::vector<CNN*> *cnns
            #if PROFILING
                ,Timer *cnnThreadTimer
            #endif
            ){
                for (int i=threadId;i<batchSize;i+=numCnnThreads) {
                    uint64_t startTime = getCurrTimeMs();
                    PlantImage* p = (*plantImages)[i].load(std::memory_order_acquire);
                    #if PROFILING
                        Timer *waitingForImageTimer = nullptr;
                        if(cnnThreadTimer){
                            waitingForImageTimer = cnnThreadTimer->addChildTimer("waitingForImage");
                        }
                    #endif
                    while (p == nullptr && (getCurrTimeMs() - startTime) < 5000){
                        //Give up if we can't get the image in 5 seconds
                        //Note: this doesn't stop the image from being loaded (if it's still loading)
                        p = (*plantImages)[i].load(std::memory_order_acquire);
                        #if DEBUG
                            std::cout << "CNN thread "<<threadId << " waiting" << std::endl;
                        #endif
                        std::this_thread::sleep_for(std::chrono::milliseconds(10)); //10ms
                    }
                    #if PROFILING
                        if(cnnThreadTimer) waitingForImageTimer->stop();
                    #endif
                    if(p!=nullptr && p->index!=-1 && p->label.length()>0){
                        (*cnns)[threadId]->backwards(p->data,p->label
                        #if PROFILING
                            ,cnnThreadTimer?cnnThreadTimer:nullptr
                        #endif
                        );
                    }
                    else missedCount.fetch_add(1, std::memory_order_relaxed);
                    if(p!=nullptr){
                        (*plantImages)[i].store(nullptr, std::memory_order_release);
                        delete p;
                    }
                    //Sometimes we won't actually do the batch size but it's only a (relatively) arbitrary number
                }
                #if PROFILING
                    if(cnnThreadTimer) cnnThreadTimer->stop();
                #endif 
            },cT,batchSize,numCnnThreads,&plantImages,d,&cnns
            #if PROFILING
                ,parentTimer?batchTimer->addChildTimer("cnnThread"+std::to_string(cT)):nullptr
            #endif
        );
    }
    int i=0;
    for(std::thread& imageThread:imageThreads){
         //No easy way to kill a thread which calls a blocking external function (without processes)
        //and so we can't have a timeout
        imageThread.join();
        #if DEBUG
            std::cout << "Image thread: "+std::to_string(i)+" joined" << std::endl;
        #endif 
        i++;
    }
    i=0;
    for(std::thread& cnnThread:cnnThreads){
        cnnThread.join();
        #if DEBUG
            std::cout << "CNN thread: "+std::to_string(i)+" joined" << std::endl;
        #endif
        i++;
    }
    n->applyGradients(cnns,batchSize
    #if PROFILING
        ,parentTimer?batchTimer:nullptr
    #endif
    );
    for(i=0;i<batchSize;i++){
        PlantImage *p = plantImages[i].load(std::memory_order_acquire);
        if(p!=nullptr){
            plantImages[i].store(nullptr, std::memory_order_release);
            delete p;
        }
    }
    #if PROFILING
        if(parentTimer) batchTimer->stop();
    #endif
}

