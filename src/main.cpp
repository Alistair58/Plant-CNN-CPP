#include <string>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <unistd.h>
#include <thread>
#include "cnn.hpp"
#include "cnnutils.hpp"
#include "dataset.hpp"
#include "plantimage.hpp"
#include "globals.hpp"
#include "utils.hpp"
#include "tensor.hpp"

//The only include with the macro defined
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

//Default values
static float LR = 0.00004f;
static int batchSize = 64;
#define TRAIN 1
#define TEST 2

std::string currDir = std::filesystem::current_path().string();
std::string datasetDirPath = "C:/Users/Alistair/Pictures/house_plant_species";
const std::string ANSI_RED = "\u001B[31m";
const std::string ANSI_RESET = "\u001B[0m";
const std::string ANSI_GREEN = "\u001B[32m";
int missedCount = 0;

static void compressionTest(Dataset *d,CNN *cnn,std::string fname);
static void trainBatch(CNN *n, Dataset *d, int batchSize,int numImageThreads,std::vector<CNN*>& cnns);
static void train(CNN *n, Dataset *d, int numBatches,int batchSize,int numImageThreads, int numCnnThreads);
static void test(CNN *n, Dataset *d, int numTest);

//TODO
//Try compression test
//Change argv parsing so that arguments are labelled (more future proof arguments) and add restart
//Speed up - see log.txt

int main(int argc,char **argv){
    //("train"|"test") 
    //"train" ->       {numBatches} {batchSize}? {LR}?
    //"test"  ->       {numTestImages} {LR}?
    Dataset *d = new Dataset(datasetDirPath,0.8f);
    const int numImageThreads = 1;
    const int numCnnThreads = 8;
    int mode = -1;
    int numBatches = -1;
    bool restart = false;
    CNN *cnn = new CNN(LR,d,restart);
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
            throw std::invalid_argument("For train, argument 2 must be the number of batches (an non-zero positive integer)");
        }
        if(argc==5){
            batchSize = atoi(argv[3]);
            if(batchSize<=0){
                throw std::invalid_argument("For train, argument 3 (if 4 arguments are provided) must be the batch size (an non-zero positive integer)");
            }
            LR = atof(argv[4]);
            if(LR<=0.0f){
                throw std::invalid_argument("For train, argument 4 (if present) must be the learning rate (a positive float)");
            }
        }
        else if(argc==4){
            float mysteryArg = atof(argv[3]);
            if(mysteryArg<=0.0f){
                throw std::invalid_argument("For train, argument 3 (if present and only 3 arguments are provided) must either be the batch size (an non-zero positive integer) or the learning rate (a positive float)");
            }
            if(mysteryArg-((int)mysteryArg)==0.0f){ //i.e. an integer
                //this must be the batch size
                //if the LR is a integer, that's just silly
                batchSize = (int) mysteryArg;
            }
            else{ //i.e. a float
                LR = mysteryArg;
            }
        }
        train(cnn,d,numBatches,batchSize,numImageThreads,numCnnThreads);
    }
    if(mode==TEST){
        int numTestImages = atoi(argv[2]);
        if(numTestImages<=0){
            throw std::invalid_argument("For test, argument 2 must be the number of test images (an non-zero positive integer)");
        }
        if(argc==4){
            LR = atof(argv[3]);
            if(LR<=0.0f){
                throw std::invalid_argument("For test, argument 3 (if present) must be the learning rate (a positive float)");
            }
        }
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
        cnns[i] = new CNN(n,LR,d,false); //shallow copy
    }
    for(int i=0;i<numBatches;i++) { // numBatches of batchSize
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
    std::cout << "Missed: "+std::to_string(missedCount) << std::endl;
    //start at 1 as we don't want to delete the original CNN (at index 0)
    for(int i=1;i<cnns.size();i++){
        delete cnns[i];
    }
}


static void trainBatch(CNN *n, Dataset *d, int batchSize,int numImageThreads,std::vector<CNN*>& cnns){ //batch size must be a multiple of numThreads
    int numCnnThreads = cnns.size();
    std::vector<std::thread> cnnThreads(numCnnThreads);
    std::vector<std::thread> imageThreads(numImageThreads);
    std::vector<PlantImage*> plantImages(batchSize);
    for(int iT=0;iT<numImageThreads;iT++){
        imageThreads[iT] = std::thread(
            [](int threadId,int batchSize,int numImageThreads,std::vector<PlantImage*> *plantImages,Dataset *d){
                for(int i=threadId;i<batchSize;i+=numImageThreads){
                    (*plantImages)[i] = d->randomImage(false);
                }
            },iT,batchSize,numImageThreads,&plantImages,d
        );
    }
    for(int cT=0;cT<numCnnThreads;cT++){ 
        cnnThreads[cT]= std::thread(
            [](int threadId,int batchSize,int numCnnThreads,std::vector<PlantImage*> *plantImages,Dataset *d,std::vector<CNN*> *cnns){
                for (int i=threadId;i<batchSize;i+=numCnnThreads) {
                    uint64_t startTime = getCurrTimeMs();
                    while((*plantImages)[i]==nullptr && (getCurrTimeMs()-5000)<startTime){
                        //Give up if we can't get the image in 5 seconds
                        //Note: this doesn't stop the image from being loaded (if it's still loading)
                        usleep(10000); //10ms
                    }
                    if((*plantImages)[i]!=nullptr && (*plantImages)[i]->index!=-1 && (*plantImages)[i]->label.length()>0){
                        (*cnns)[threadId]->backwards((*plantImages)[i]->data,(*plantImages)[i]->label);
                        delete (*plantImages)[i];
                        (*plantImages)[i] = nullptr;
                    }
                    else missedCount++;
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
    //We did a shallow copy of the gradients and so the accumulation of all the CNN gradients will be in the original CNN's gradient
    n->applyGradients();
    for(PlantImage *ptr:plantImages){
        if(ptr!=nullptr) delete ptr;
    }
}

static void compressionTest(Dataset *d,CNN *cnn,std::string fname){
    PlantImage testing = 
    (fname.length()==0)? d->randomImageObj(false) : PlantImage(fname, "");
    Tensor img = cnn->parseImg(testing.data);
    std::vector<int> mapDimens = cnn->getMapDimens();
    unsigned char *data = new unsigned char[mapDimens[0]*mapDimens[0]*3];
    std::vector<int> imgDimens = img.getDimens();
    int height = imgDimens[1];
    int width = imgDimens[2];
    for(int y=0;y<height;y++){
        for(int x=0;x<width;x++){
            int i = (y*width + x)*3;
            data[i] = (int)*img[{0,y,x}];
            data[i+1] = (int)*img[{1,y,x}];
            data[i+2] = (int)*img[{2,y,x}];
        }
    }
    if(!stbi_write_jpg((currDir+"/testing.jpg").c_str(),width,height,3,data,width*height*3)){
        std::cerr << "Could not save image\n";
    }
    else {
        std::cout << "Saved testing.jpg\n";
    }
    std::cout << testing.label+" "+std::to_string(testing.index) << std::endl;
    delete[] data;
}

