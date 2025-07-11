#include <string>
#include <filesystem>
#include <iostream>
#include <chrono>
#include "cnn.hpp"
#include "cnnutils.hpp"
#include "dataset.hpp"
#include "plantimage.hpp"
#include "globals.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static float LR = 0.00004f;
std::filesystem::path currDir = std::filesystem::current_path();
std::string datasetDirPath = "C:/Users/Alistair/Pictures/house_plant_species";
const std::string ANSI_RED = "\u001B[31m";
const std::string ANSI_RESET = "\u001B[0m";
const std::string ANSI_GREEN = "\u001B[32m";
int missedCount = 0;

//TODO
//Change the matrix implementation to be continguous in memory
//Probably should call it tensor
//And look at ChatGPT example

int main(int argc,char **argv){
    Dataset *d = new Dataset(datasetDirPath);
    CNN *cnn = new CNN(LR,d,false);
    train(cnn,d,500,64,4,4); 
    test(cnn,d,1000);
    delete d;
    delete cnn;
}
    
static void test(CNN *n, Dataset *d, int numTest) {
    int correctCount = 0;
    for (int i = 0; i < numTest; i++) {
        PlantImage pI = d.randomImage(true);
        if(pI.label.length()==0){
            std::string response = n.forwards(pI.data);
            bool correct = response==pI.label;
            std::cout << (((correct)?ANSI_GREEN:ANSI_RED)
            +pI.label +" ("+std::to_string(pI.index)+ ") Computer said: " + response+ANSI_RESET) << std::endl;
            if(correct) correctCount++;
        }
        else i--;
    }
    std::cout << ("Accuracy: "+std::to_string((float)correctCount*100/numTest)+"%") <<std::endl;
}

static void train(CNN *n, Dataset *d, int numBatches,int batchSize,int numImageThreads, int numCnnThreads) {
    std::chrono::milliseconds startTime = duration_cast< milliseconds >(
        system_clock::now().time_since_epoch()
    );
    int missedCount = 0;
    for (int i = 0; i < numBatches; i++) { // numBatches of batchSize
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
    std::chrono::milliseconds endTime = duration_cast< milliseconds >(
        system_clock::now().time_since_epoch()
    );
    int secs = (int)((endTime-startTime)/1000);
    int mins = (int) (secs/60);
    int hours = (int) (mins/60);
    std::cout << "Took: "+hours+" hr(s) "+mins%60+" min(s) "+secs%60+" sec(s)" << std::endl;
    std::cout << "Missed: "+String.valueOf(missedCount) << std::endl;
}


static void trainBatch(CNN *n, Dataset *d, int batchSize,int numImageThreads, int numCnnThreads) { //batch size must be a multiple of numThreads
    std::vector<CNN*> cnns(numCnnThreads);
    std::vector<std::thread> cnnThreads(numCnnThreads);
    std::vector<std::thread> imageThreads(numImageThreads);
    std::vector<PlantImage*> plantImages(batchSize);
    for(int iT=0;iT<numImageThreads;iT++){
        imageThreads[iT] = std::thread(
            [](int threadId){
                for(int i=threadId;i<batchSize;i+=numImageThreads){
                    plantImages[i] = d.randomImage(false);
                }
            },iT 
        );
    }
    cnns[0] = n;
    for(int cT=0;cT<numCnnThreads;cT++){
        if(cT>0){
            cnns[cT] = new CNN(n,LR,d);
        }
        cnnThreads[cT]= std::thread(
            [](int threadId){
                for (int i=threadId;i<batchSize;i+=numCnnThreads) {
                    long startTime = getCurrTime();
                    while(nullptr == plantImages[i] && getCurrTime()-5000<startTime){
                        //Give up if we can't get the image in 5 seconds
                        //Note: this doesn't stop the image from being loaded (if it's still loading)
                        usleep(10);
                    }
                    if(nullptr != plantImages[i] && plantImages[i]->label.length()>0) cnns[threadId].backwards(plantImages[i]->data,plantImages[i]->label);
                    else missedCount++;
                    //Sometimes we won't actually do the batch size but it's only a (relatively) arbitrary number
                }
            },cT
        );
    }
    for(int t=0;t<max(numCnnThreads,numImageThreads);t++){
        if(t<numImageThreads){
            join(&imageThreads[t],10);
            
        }
        if(t<numCnnThreads){
            join(&cnnThreads[t],10);

    }
    n->applyGradients(cnns);
}

static void compressionTest(Dataset *d,CNN *cnn,std::string fname){
    PlantImage *testing = 
    (nullptr == fname)? d->randomImage(false) : new PlantImage(fname, "");
    Tensor img = cnn->parseImg(testing.data);
    unsigned char *data = new unsigned char[cnn->mapDimens[0]*cnn->mapDimens[0]*3];
    for(int y=0;y<height;y++){
        for(int x=0;x<width;x++){
            int i = (y*width + x)*3;
            data[i] = img[0][y][x];
            data[i+1] = img[1][y][x];
            data[i+2] = img[2][y][x];
        }
    }
    if(!stbi_write_jpg(currDir+"/plantcnn/testing.jpg",width,height,3,data,width*height*3)){
        std::cerr << "Could not save image\n";
    }
    else {
        std::cout << "Saved testing.jpg\n";
    }
    std::cout << testing.label+" "+testing.index << std::endl;
    delete[] data;
}

