#include <string>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <unistd.h>
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

//The only include with the macro defined
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

//Default values
static float LR = 1.5625e-4f;
static int batchSize = 64;
#define TRAIN 1
#define TEST 2

std::string currDir = std::filesystem::current_path().string();
std::string datasetDirPath = "C:/Users/Alistair/Pictures/house_plant_species";
const std::string ANSI_RED = "\u001B[31m";
const std::string ANSI_RESET = "\u001B[0m";
const std::string ANSI_GREEN = "\u001B[32m";
std::atomic<int> missedCount{0};

static void compressionTest(Dataset *d,CNN *cnn,std::string fname);
static void trainBatch(CNN *n, Dataset *d, int batchSize,int numImageThreads,std::vector<CNN*>& cnns);
static void train(CNN *n, Dataset *d, int numBatches,int batchSize,int numImageThreads, int numCnnThreads);
static void test(CNN *n, Dataset *d, int numTest);


//TODO
//Speed up - see log.txt

int main(int argc,char **argv){
    //("train"|"test") 
    //"train" ->       {numBatches} (rs=(true|false))? (bs={batchSize})? (lr={LR})?
    //"test"  ->       {numTestImages} (lr={LR})?
    Dataset *d = new Dataset(datasetDirPath,0.8f);
    CNN *cnn = nullptr;
    const int numImageThreads = 2;
    const int numCnnThreads = 8;
    int mode = -1;
    int numBatches = -1;
    bool restart = false;
    if(argc<3){
        throw std::invalid_argument("argv must contain at least 2 arguments");
    }
    if(stricmp(argv[1],"train")==0) mode = TRAIN;
    else if(stricmp(argv[1],"test")==0) mode = TEST;
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
                    restart = true;
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
        cnn = new CNN(LR,d,restart);
        train(cnn,d,numBatches,batchSize,numImageThreads,numCnnThreads);
    }
    if(mode==TEST){
        int numTestImages = atoi(argv[2]);
        if(numTestImages<=0){
            throw std::invalid_argument("For test, argument 2 must be the number of test images (a positive integer)");
        }
        if(argc==4){
            std::vector<std::string> splitRes = strSplit(argv[3],{'='});
            if(splitRes.size()!=2){
                throw std::invalid_argument("Optional arguments must be in the format {parameter}={value}");
            }
            if(toLower(splitRes[0])!="lr"){
                throw std::invalid_argument("Optional parameter \""+splitRes[0]+"\" is invalid for test");
            }
            LR = stof(splitRes[1]);
            if(LR<=0.0f){
                throw std::invalid_argument("Parameter \"lr\" (learning rate) must be a positive float");
            }
        }
        cnn = new CNN(LR,d,false);
        test(cnn,d,numTestImages);
    }
    delete d;
    delete cnn;
}
    
static void test(CNN *n, Dataset *d, int numTest){
    int correctCount = 0;
    for (int i=0;i<numTest;i++) {
        PlantImage pI = d->randomImageObj(true);
        if(pI.label.length()!=0){
            std::string response = n->forwards(pI.data);
            bool correct = response==pI.label;
            std::cout << (((correct)?ANSI_GREEN:ANSI_RED) +
            pI.label +" ("+std::to_string(pI.index)+ ") Computer said: " + response+ANSI_RESET) << std::endl;
            if(correct) correctCount++;
        }
        else i--;
    }
    std::cout << ("Accuracy: "+std::to_string((float)correctCount*100/numTest)+"%") <<std::endl;
}

static void train(CNN *n, Dataset *d, int numBatches,int batchSize,int numImageThreads, int numCnnThreads){
    uint64_t startTime = getCurrTimeMs();
    std::vector<CNN*> cnns(numCnnThreads);
    cnns[0] = n;
    for(int i=1;i<numCnnThreads;i++){
        //shallow copy of weights and kernels
        //Must be shallow as apply gradients only updates cnn[0]'s weights and kernels
        cnns[i] = new CNN(n,LR,d,false); 
    }
    for(int i=0;i<numBatches;i++){ // numBatches of batchSize
        trainBatch(n, d, batchSize,numImageThreads,cnns);
        if(i%10 == 0 && i>0){ //save every 10 batches
            n->saveKernels();
            n->saveWeights();
            std::cout << "Saved" << std::endl;
        }
        std::cout << i << std::endl;
    }
    n->saveWeights();
    n->saveKernels();
    std::cout << "Done" << std::endl;
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
}


static void trainBatch(CNN *n, Dataset *d, int batchSize,int numImageThreads,std::vector<CNN*>& cnns){ //batch size must be a multiple of numThreads
    int numCnnThreads = cnns.size();
    std::vector<std::thread> cnnThreads(numCnnThreads);
    std::vector<std::thread> imageThreads(numImageThreads);
    std::vector<std::atomic<PlantImage*>> plantImages(batchSize);
    for(int i=0;i<batchSize;i++) plantImages[i].store(nullptr,std::memory_order_relaxed);
    for(int iT=0;iT<numImageThreads;iT++){
        imageThreads[iT] = std::thread(
            [](int threadId,int batchSize,int numImageThreads,std::vector<std::atomic<PlantImage*>> *plantImages,Dataset *d){
                for(int i=threadId;i<batchSize;i+=numImageThreads){
                    PlantImage *p = d->randomImage(false);
                    (*plantImages)[i].store(p,std::memory_order_release); 
                }
            },iT,batchSize,numImageThreads,&plantImages,d
        );
    }
    for(int cT=0;cT<numCnnThreads;cT++){ 
        cnnThreads[cT]= std::thread(
            [](int threadId,int batchSize,int numCnnThreads,std::vector<std::atomic<PlantImage*>> *plantImages,Dataset *d,std::vector<CNN*> *cnns){
                for (int i=threadId;i<batchSize;i+=numCnnThreads) {
                    uint64_t startTime = getCurrTimeMs();
                    PlantImage* p = (*plantImages)[i].load(std::memory_order_acquire);
                    while (p == nullptr && (getCurrTimeMs() - startTime) < 5000){
                        //Give up if we can't get the image in 5 seconds
                        //Note: this doesn't stop the image from being loaded (if it's still loading)
                        p = (*plantImages)[i].load(std::memory_order_acquire);
                        usleep(10000); //10ms
                    }
                    if(p!=nullptr && p->index!=-1 && p->label.length()>0){
                        (*cnns)[threadId]->backwards(p->data,p->label);
                    }
                    else missedCount.fetch_add(1, std::memory_order_relaxed);
                    if(p!=nullptr){
                        (*plantImages)[i].store(nullptr, std::memory_order_release);
                        delete p;
                    }
                    //Sometimes we won't actually do the batch size but it's only a (relatively) arbitrary number
                }
            },cT,batchSize,numCnnThreads,&plantImages,d,&cnns
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
    n->applyGradients(cnns);
    for(i=0;i<batchSize;i++){
        PlantImage *p = plantImages[i].load(std::memory_order_acquire);
        if(p!=nullptr){
            plantImages[i].store(nullptr, std::memory_order_release);
            delete p;
        }
    }
}

