#include<iostream>
#include<string>
#include<vector>
#include<unordered_map>
#include<cstdlib>
#include<filesystem>
namespace fs = std::filesystem;
#include"dataset.hpp"
#include <nlohmann/json.hpp>
#include "globals.hpp"

class Dataset {
    private:
        std::vector<int> indices;
        int size = 0; 
        std::string dirPath;
        std::vector<std::string> fileExtensions; //Includes the dots
        std::vector<std::vector<float>> pixelStats;

        void loadPixelStats(){
            try{
                std::ifstream statsFile(Main.currDir+"/res/stats.json");
                nlohmann::json::json jsonStats;
                statsFile >> jsonStats;
                statsFile.close();

                if(jsonStats.size()!=3) throw InputMismatchException();
                pixelStats = jsonStats.get<std::vector<std::vector<float>>>();
                if((int)pixelStats[2][0]!=this.size){
                    std::cout << 
                    (Main.ANSI_RED+"Stats file mismatch warning:\n\tThe dataset has "
                    +String.valueOf(size)+" items and the stats file has "+
                    String.valueOf((int)pixelStats[2][0])+" items"+Main.ANSI_RESET)
                    << std::endl;
                }
            } 
            catch(const InputMismatchException& e){
                std::cout << (ANSI_RED+"Stats file is not in the format {{mean1,..},{stdDev1,..},{count}}"+ANSI_RESET) << std::endl;
                exit(1);
            }
        }   
    public:
        std::vector<std::string> plantNames;
        std::unordered_map<std::string,int> plantToIndex;

        Dataset(std::string dirPathInp){
            this->dirPath = dirPathInp;
            int i = 0;
            String fileExtension;
            for(const auto &entry:fs::directory_iterator(dirPath)){
                if(fs::isDirectory(entry)){
                    indices.push_back(i);
                    plantNames.add(entry.filename());
                    for(const auto &subEntry:fs::directory_iterator(entry)){
                        std::string fileExtension = regex_replace(subEntry.filename(),"^[^\\.]+", "");
                        if(find(fileExtensions.begin(),fileExtensions.end(),fileExtension)==fileExtensions.end()){
                            fileExtensions.push_back(fileExtension);
                        }
                        i++;
                    }
                }
            }
            this->size = i;
            std::cout << "Dataset has "+size+" items" << std::endl;
        }

        float *getPixelMeans(){
            return this->pixelStats[0];
        }

        float *getPixelStdDevs(){
            return this->pixelStats[1];
        }

        PlantImage *randomImage(bool test){
            int index = rand() % this->size;
            int subIndex,startIndex,prevIndex = this->size;
            std::string fname,plantName;
            for(int i=indices.size()-1;i>=0;i--){ //iterate through plant classes
                startIndex = indices[i];
                if(index >= startIndex){ //until the random number is in this category
                    subIndex = index - startIndex;
                    int categorySize = prevIndex - startIndex;
                    if(test && subIndex<0.8*categorySize){ //80 20 split
                        subIndex = ((int)(0.8*categorySize)) +(rand()%(categorySize+1)-(int)(0.8*categorySize)) //indices start at 1 so we have an extra at the end
                    }
                    else if(!test && subIndex>0.8*categorySize){ //training
                        subIndex = 1 + (rand()%(0.8*categorySize)); //image names start at 1.jpg
                    }
                    plantName = plantNames[i];
                    fname = dirPath+"/"+plantName+"/"+std::to_string(subIndex);
                    for(std::string fileExtension:fileExtensions){ //Try all file extensions
                        try {
                            PlantImage *plantImage = new PlantImage(fname+fileExtension,plantName);
                            if(plantImage->data[0][0][0] > 0){ //valid image
                                return plantImage;
                            }
                            delete plantImage;
                        } catch (Exception e) {
                            System.out.println(e.toString());
                            //Trying to get the dimensions of not an image will produce an NPE 
                        }
                    } 
                    break;
                }
                prevIndex = startIndex;
            }
            return new PlantImage("","");
        }

        void compilePixelStats(){
            json stats = {{0,0,0},{0,0,0},{0}};
            int numPlants = 0;
            for(int i=0;i<indices.size();i++){
                String plantName = plantNames[i];
                std::cout << "\n"+plantName+"\n" << std::endl;
                int categorySize = (i+1==indices.size()?size:indices.get(i+1))-indices.get(i);
                for(int j=1;j<=categorySize;j++){ //all the plants in this category (indices start at 1)
                    std::cout << plantName+" "+std::to_string(j) << std::endl;
                    std::string fname = dirPath+"/"+plantName+"/"+std::string(j);
                    PlantImage *plantImage;
                    for(String fileExtension:fileExtensions){ //Try all file extensions
                        plantImage = new PlantImage(fname+fileExtension,plantName);
                        if(plantImage->data.size()==0 || plantImage->data[0].size()==0 ||
                        plantImage->data[0][0].size()==0){
                            continue;
                        }
                        if(plantImage->data[0][0][0] > 0){
                            break;
                        }
                    } 
                    if(plantImage==nullptr || plantImage.data.length<3) continue;
                    double[][] imageStats = new double[][] {{0,0,0},{0,0,0}}; //means, stdDevs
                    for(int c=0;c<3;c++){
                        for(int y=0;y<plantImage.data[c].size();y++){
                            for(int x=0;x<plantImage.data[c][y].size();x++){
                                imageStats[0][c] += plantImage.data[c][y][x]; //add to mean
                                imageStats[1][c] += plantImage.data[c][y][x]*plantImage.data[c][y][x]; //add to stdDev
                            }
                        }
                        //add to total
                        float numPixels = plantImage.data[c].size()*plantImage.data[c][0].size();
                        float mean = (float) (imageStats[0][c]/(numPixels));
                        float stdDev = (float) sqrt(imageStats[1][c]/(numPixels) - (mean*mean));
                        if(std::isnan(stdDev) || (imageStats[1][c]/(numPixels))<(mean*mean)){
                            std::cout << "Error" << std::endl;
                        }
                        else{
                            stats[0][c] += mean;
                            stats[1][c] += stdDev;
                        }
                        
                    }
                }
                
                //save after each type of plant
                numPlants+=categorySize;
                for(int c=0;c<3;c++){
                    stats[0][c] /= numPlants;
                    stats[1][c] /= numPlants;
                }
                stats[2][0] = numPlants;
                std::ofstream statsFile(currDir+"/res/stats.json");
                statsFile << stats.dump();
                statsFile.close();
                for(int c=0;c<3;c++){ //resume
                    stats[0][c] *= numPlants;
                    stats[1][c] *= numPlants;
                }
            }
        }

};
