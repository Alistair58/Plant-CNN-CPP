#include<string>
#include<filesystem>
#include<iostream>
#include<chrono>
#include"cnn.hpp"
#include"cnnutils.hpp"
#include"dataset.hpp"
#include"plantimage.hpp"
#include "globals.hpp"

static float LR = 0.00004f;
std::filesystem::path currDir = std::filesystem::current_path();
std::string datasetDirPath = "C:/Users/Alistair/Pictures/house_plant_species";
const std::string ANSI_RED = "\u001B[31m";
const std::string ANSI_RESET = "\u001B[0m";
const std::string ANSI_GREEN = "\u001B[32m";


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
    CNN* cnns[numCnnThreads];
    std::thread cnnThreads[numCnnThreads];
    std::thread imageThreads[numImageThreads];
    PlantImage* plantImages[batchSize];
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
        final int cnnThreadId = cT; //has to be final
        Thread cnnThread = new Thread(new Runnable(){
            public int threadId = cnnThreadId;
            @Override
            public void run(){
                for (int i=threadId;i<batchSize;i+=numCnnThreads) {
                    long startTime = System.currentTimeMillis();
                    while(null == plantImages[i] && System.currentTimeMillis()-5000<startTime){
                        //Give up if we can't get the image in 5 seconds
                        //Note: this doesn't stop the image from being loaded (if it's still loading)
                        try {
                            Thread.sleep(10);
                        } catch (InterruptedException e) {
                        }
                    }
                    if(null != plantImages[i] && !"".equals(plantImages[i].label)) cnns[threadId].backwards(plantImages[i].data,plantImages[i].label);
                    else missedCount++;
                    //Sometimes we won't actually do the batch size but it's only a (relatively) arbitrary number
                }
            }
        });
        cnnThreads[cT] = cnnThread;
        cnnThread.start();
    }
    for(int t=0;t<Math.max(numCnnThreads,numImageThreads);t++){
        try { 
            auto future = std::async(std::launch::async, &std::thread::join, &imageThreads[t]);
            if (future.wait_for(std::chrono::seconds(10)) //10s max wait
                == std::future_status::timeout) {
                    std::terminate()
            }
            if(t<numImageThreads) .join(10000); 
            if(t<numCnnThreads) cnnThreads[t].join(10000);
        } catch (InterruptedException e) {
            System.out.println(e);
        }
    }
    n->applyGradients(cnns);
}

static void compressionTest(Dataset d,CNN cnn,String fname){
    PlantImage testing = 
    (null == fname)? d.randomImage(false) : new PlantImage(fname, "");
    float[][][] img = cnn.parseImg(testing.data);
    File outputfile = new File(currDir+"/plantcnn/testing.jpg");
    BufferedImage bufferedImage = new BufferedImage(cnn.mapDimens[0], cnn.mapDimens[0], BufferedImage.TYPE_INT_RGB);
    Color colour;
    for (int i = 0; i < img[0].length; i++) {
        for (int j = 0; j < img[0][i].length; j++) {
            colour = new Color(img[0][i][j]/255,img[1][i][j]/255,img[2][i][j]/255); //Color wants floats between 0 and 1
            bufferedImage.setRGB(j, i,colour.getRGB() );

        }
    }
    try {
        ImageIO.write(bufferedImage, "jpg", outputfile);
    } catch (IOException e) {
        System.out.println(e);
    }
    System.out.println(testing.label+" "+testing.index);
}
