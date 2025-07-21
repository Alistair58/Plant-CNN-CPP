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
#include <unistd.h>

static float LR = 0.00004f;
std::string currDir = std::filesystem::current_path().string();
std::string datasetDirPath = "C:/Users/Alistair/Pictures/house_plant_species";
const std::string ANSI_RED = "\u001B[31m";
const std::string ANSI_RESET = "\u001B[0m";
const std::string ANSI_GREEN = "\u001B[32m";
int missedCount = 0;

//TODO
//Code review:

//tensor.hpp _/
//tensor.cpp _/
//plantimage.hpp _/
//plantimage.cpp _/
//globals.hpp
//dataset.hpp
//dataset.cpp
//cnnutils.hpp
//cnnutils.cpp
//cnn.hpp
//cnn.cpp 

int main(int argc,char **argv){
    std::string testStr1 = "this.is.a.test.String";
    std::vector<std::string> res1 = strSplit(testStr1,{'.'});
    for(std::string str:res1){
        std::cout << str+" ";
    }
    std::cout << "\n";
    std::string testStr2 = "..this.is.a.test.Strin.g..";
    std::vector<std::string> res2 = strSplit(testStr2,{'.'});
    for(std::string str:res2){
        std::cout << str+" ";
    }
    // Dataset *d = new Dataset(datasetDirPath);
    // CNN *cnn = new CNN(LR,d,false);
    // train(cnn,d,500,64,4,4); 
    // test(cnn,d,1000);
    // delete d;
    // delete cnn;
}
    
static void test(CNN *n, Dataset *d, int numTest) {
    int correctCount = 0;
    for (int i = 0; i < numTest; i++) {
        PlantImage *pI = d->randomImage(true);
        if(pI->label.length()==0){
            std::string response = n->forwards(pI->data);
            bool correct = response==pI->label;
            std::cout << (((correct)?ANSI_GREEN:ANSI_RED)
            +pI->label +" ("+std::to_string(pI->index)+ ") Computer said: " + response+ANSI_RESET) << std::endl;
            if(correct) correctCount++;
        }
        else i--;
    }
    std::cout << ("Accuracy: "+std::to_string((float)correctCount*100/numTest)+"%") <<std::endl;
}

static void train(CNN *n, Dataset *d, int numBatches,int batchSize,int numImageThreads, int numCnnThreads) {
    long startTime = getCurrTime();
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
    long endTime = getCurrTime();
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


static void trainBatch(CNN *n, Dataset *d, int batchSize,int numImageThreads, int numCnnThreads) { //batch size must be a multiple of numThreads
    std::vector<CNN*> cnns(numCnnThreads);
    std::vector<std::thread> cnnThreads(numCnnThreads);
    std::vector<std::thread> imageThreads(numImageThreads);
    std::vector<PlantImage*> plantImages(batchSize);
    for(int iT=0;iT<numImageThreads;iT++){
        imageThreads[iT] = std::thread(
            [](int threadId,int batchSize,int numImageThreads,std::vector<PlantImage*> plantImages,Dataset *d){
                for(int i=threadId;i<batchSize;i+=numImageThreads){
                    plantImages[i] = d->randomImage(false);
                }
            },iT,batchSize,numImageThreads,plantImages,d
        );
    }
    cnns[0] = n;
    for(int cT=0;cT<numCnnThreads;cT++){
        if(cT>0){
            cnns[cT] = new CNN(n,LR,d);
        }
        cnnThreads[cT]= std::thread(
            [](int threadId,int batchSize,int numCnnThreads,std::vector<PlantImage*> plantImages,Dataset *d,std::vector<CNN*> cnns){
                for (int i=threadId;i<batchSize;i+=numCnnThreads) {
                    long startTime = getCurrTime();
                    while(plantImages[i]->index==-1 && getCurrTime()-5000<startTime){
                        //Give up if we can't get the image in 5 seconds
                        //Note: this doesn't stop the image from being loaded (if it's still loading)
                        usleep(10000); //10ms
                    }
                    if(plantImages[i]->index!=-1 && plantImages[i]->label.length()>0) cnns[threadId]->backwards(plantImages[i]->data,plantImages[i]->label);
                    else missedCount++;
                    //Sometimes we won't actually do the batch size but it's only a (relatively) arbitrary number
                }
            },cT,batchSize,numCnnThreads,plantImages,d,cnns
        );
    }
    for(int t=0;t<std::max(numCnnThreads,numImageThreads);t++){
        if(t<numImageThreads){
            join(&imageThreads[t],10);
        }
        if(t<numCnnThreads){
            join(&cnnThreads[t],10);
        }
    }
    n->applyGradients(cnns);
    for(PlantImage *ptr:plantImages){
        delete ptr;
    }
    for(CNN *ptr:cnns){
        delete ptr;
    }
}

static void compressionTest(Dataset *d,CNN *cnn,std::string fname){
    PlantImage *testing = 
    (fname.length()==0)? d->randomImage(false) : new PlantImage(fname, "");
    Tensor img = cnn->parseImg(testing->data);
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
    if(!stbi_write_jpg((currDir+"/plantcnn/testing.jpg").c_str(),width,height,3,data,width*height*3)){
        std::cerr << "Could not save image\n";
    }
    else {
        std::cout << "Saved testing.jpg\n";
    }
    std::cout << testing->label+" "+std::to_string(testing->index) << std::endl;
    delete[] data;
}

