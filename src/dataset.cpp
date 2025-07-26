#include "dataset.hpp"


void Dataset::loadPixelStats(){
    std::ifstream statsFile(currDir+"/res/stats.json");
    nlohmann::json jsonStats;
    statsFile >> jsonStats;
    statsFile.close();

    if(jsonStats.size()!=3){
        throw std::invalid_argument(ANSI_RED+"Stats file is not in the format {{mean1,..},{stdDev1,..},{count}}"+ANSI_RESET);
    }
    pixelStats = jsonStats.get<d2>();
    if((int)pixelStats[2][0]!=this->size){
        std::cout << 
        (ANSI_RED+"Stats file mismatch warning:\n\tThe dataset has "
        +std::to_string(size)+" items and the stats file has "+
        std::to_string((int)pixelStats[2][0])+" items"+ANSI_RESET)
        << std::endl;
    }
}   

Dataset::Dataset(std::string dirPathInp){
    this->dirPath = dirPathInp;
    int i = 0;
    std::string fileExtension;
    for(const auto &entry:fs::directory_iterator(dirPath)){
        if(fs::is_directory(entry)){
            indices.push_back(i);
            char *folderPath = (char*) entry.path().c_str();
            std::regex parentFoldersRegex("[^/]+\\/");
            std::string folderName = std::regex_replace(folderPath,parentFoldersRegex,"");
            plantNames.push_back(folderName);
            for(const auto &subEntry:fs::directory_iterator(entry)){
                std::regex allExceptExtensionRegex("^[^\\.]+");
                char *filePath = (char*) subEntry.path().c_str();
                std::string fileExtension = std::regex_replace(filePath,allExceptExtensionRegex, "");
                if(find(fileExtensions.begin(),fileExtensions.end(),fileExtension)==fileExtensions.end()){
                    fileExtensions.push_back(fileExtension);
                }
                i++;
            }
        }
    }
    this->size = i;
    std::cout << "Dataset has "+std::to_string(size)+" items" << std::endl;
}

std::vector<float> Dataset::getPixelMeans(){
    return this->pixelStats[0];
}

std::vector<float> Dataset::getPixelStdDevs(){
    return this->pixelStats[1];
}

PlantImage *Dataset::randomImage(bool test){
    int index = rand() % this->size;
    int subIndex,startIndex,prevIndex = this->size;
    std::string fname,plantName;
    for(int i=indices.size()-1;i>=0;i--){ //iterate through plant classes
        startIndex = indices[i];
        if(index >= startIndex){ //until the random number is in this category
            subIndex = index - startIndex;
            int categorySize = prevIndex - startIndex;
            if(test && subIndex<0.8*categorySize){ //80 20 split
                subIndex = ((int)(0.8*categorySize)) +(rand()%(categorySize+1)-(int)(0.8*categorySize)); //indices start at 1 so we have an extra at the end
            }
            else if(!test && subIndex>0.8*categorySize){ //training
                subIndex = 1 + (rand()%((int)(0.8f*categorySize))); //image names start at 1.jpg
            }
            plantName = plantNames[i];
            fname = dirPath+"/"+plantName+"/"+std::to_string(subIndex);
            for(std::string fileExtension:fileExtensions){ //Try all file extensions
                PlantImage *plantImage = new PlantImage(fname+fileExtension,plantName);
                if(*(plantImage->data[0]) > 0){ //valid image
                    return plantImage;
                }
            } 
            break;
        }
        prevIndex = startIndex;
    }
    return new PlantImage("","");
}

PlantImage Dataset::randomImageObj(bool test){
    PlantImage *ptr = randomImage(test);
    PlantImage obj = *ptr;
    delete ptr;
    return obj;
}

void Dataset::compilePixelStats(){
    nlohmann::json stats = {{0,0,0},{0,0,0},{0}};
    int numPlants = 0;
    for(int i=0;i<this->indices.size();i++){
        std::string plantName = plantNames[i];
        std::cout << "\n"+plantName+"\n" << std::endl;
        int categorySize = (i+1==indices.size()?size:indices[i+1])-indices[i];
        for(int j=1;j<=categorySize;j++){ //all the plants in this category (indices start at 1)
            std::cout << plantName+" "+std::to_string(j) << std::endl;
            std::string fname = dirPath+"/"+plantName+"/"+std::to_string(j);
            PlantImage plantImage;
            std::vector<int> dataDimens;
            for(std::string fileExtension:fileExtensions){ //Try all file extensions
                plantImage = PlantImage(fname+fileExtension,plantName);
                dataDimens = plantImage.data.getDimens();
                if(plantImage.data.getTotalSize()==0 || dataDimens.size()!=3){
                    continue;
                }
                if(*plantImage.data[0] > 0){
                    break;
                }
            } 
            if(
                plantImage.data.getTotalSize()==0 || 
                plantImage.data.getDimens().size()!=3 || 
                plantImage.data.getDimens()[0]<3
            ) continue;
            std::vector<std::vector<double>> imageStats = {{0,0,0},{0,0,0}}; //means, stdDevs
            for(int c=0;c<3;c++){
                for(int y=0;y<dataDimens[1];y++){
                    for(int x=0;x<dataDimens[2];x++){
                        imageStats[0][c] += *plantImage.data[{c,y,x}]; //add to mean
                        imageStats[1][c] += (*plantImage.data[{c,y,x}]) * (*plantImage.data[{c,y,x}]); //add to stdDev
                    }
                }
                //add to total
                float numPixels = dataDimens[1]*dataDimens[2];
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
            stats[0][c] = ((double) stats[0][c])/numPlants;
            stats[1][c] = ((double) stats[1][c])/numPlants;
        }
        stats[2][0] = numPlants;
        std::ofstream statsFile(currDir+"/res/stats.json");
        statsFile << stats.dump();
        statsFile.close();
        for(int c=0;c<3;c++){ //resume
            stats[0][c] = ((double) stats[0][c])*numPlants;
            stats[1][c] = ((double) stats[1][c])*numPlants;
        }
    }
}


