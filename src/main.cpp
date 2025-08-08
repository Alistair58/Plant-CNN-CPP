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


static float LR = 0.00004f;

std::string currDir = std::filesystem::current_path().string();
std::string datasetDirPath = "C:/Users/Alistair/Pictures/house_plant_species";
const std::string ANSI_RED = "\u001B[31m";
const std::string ANSI_RESET = "\u001B[0m";
const std::string ANSI_GREEN = "\u001B[32m";
int missedCount = 0;

static void compressionTest(Dataset *d,CNN *cnn,std::string fname);
static void trainBatch(CNN *n, Dataset *d, int batchSize,int numImageThreads, int numCnnThreads);
static void train(CNN *n, Dataset *d, int numBatches,int batchSize,int numImageThreads, int numCnnThreads);
static void test(CNN *n, Dataset *d, int numTest);



//TODO
//Why is it still slower than the Java implementation
//See how many threads is optimal

int main(int argc,char **argv){
    Dataset *d = new Dataset(datasetDirPath,0.8f);
    CNN *cnn = new CNN(LR,d,true);
    train(cnn,d,4,64,1,4); 
    //train(cnn,d,500,64,4,4); 
    //test(cnn,d,1000);
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
    for (int i=0;i<numBatches;i++) { // numBatches of batchSize
        trainBatch(n, d, batchSize,numImageThreads,numCnnThreads);
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
}


static void trainBatch(CNN *n, Dataset *d, int batchSize,int numImageThreads, int numCnnThreads){ //batch size must be a multiple of numThreads
    std::vector<CNN*> cnns(numCnnThreads);
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
    cnns[0] = n;
    for(int cT=0;cT<numCnnThreads;cT++){
        if(cT>0){
            cnns[cT] = new CNN(n,LR,d,false);
        }
        cnnThreads[cT]= std::thread(
            [](int threadId,int batchSize,int numCnnThreads,std::vector<PlantImage*> *plantImages,Dataset *d,std::vector<CNN*> *cnns){
                for (int i=threadId;i<batchSize;i+=numCnnThreads) {
                    uint64_t startTime = getCurrTimeMs();
                    while((*plantImages)[i]==nullptr && (getCurrTimeMs()-30000)<startTime){
                        //Give up if we can't get the image in 30 seconds
                        //Note: this doesn't stop the image from being loaded (if it's still loading)
                        usleep(10000); //10ms
                    }
                    std::cout << "\n";
                    if((*plantImages)[i]!=nullptr && (*plantImages)[i]->index!=-1 && (*plantImages)[i]->label.length()>0){
                        (*cnns)[threadId]->backwards((*plantImages)[i]->data,(*plantImages)[i]->label);
                    }
                    else missedCount++;
                    //Sometimes we won't actually do the batch size but it's only a (relatively) arbitrary number
                }
            },cT,batchSize,numCnnThreads,&plantImages,d,&cnns
        );
    }
    for(int t=0;t<std::max(numCnnThreads,numImageThreads);t++){
        if(t<numImageThreads){
            //No easy way to kill a thread which calls a blocking external function (without processes)
            //and so we can't have a timeout
            imageThreads[t].join();
            #if DEBUG
                std::cout << "Image thread: "+std::to_string(t)+" joined" << std::endl;
            #endif 

        }
        if(t<numCnnThreads){
            cnnThreads[t].join();
            #if DEBUG
                std::cout << "CNN thread: "+std::to_string(t)+" joined" << std::endl;
            #endif
        }
    }
    //We did a shallow copy of the gradients and so the accumulation of all the CNN gradients will be in the original CNN's gradient
    n->applyGradients();
    for(PlantImage *ptr:plantImages){
        delete ptr;
    }
    //start at 1 as we don't want to delete the original CNN (at index 0)
    for(int i=1;i<cnns.size();i++){
        delete cnns[i];
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

