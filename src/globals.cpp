#include "globals.hpp"
//The only include with the macro defined
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

std::string currDir = std::filesystem::current_path().string();
std::string datasetDirPath = "/home/alistair/Pictures/house_plant_species";

const std::string ANSI_RED = "\u001B[31m";
const std::string ANSI_RESET = "\u001B[0m";
const std::string ANSI_GREEN = "\u001B[32m";
const std::string ANSI_CLEAR_LINE = "\033[2K";
//I don't want black or white
const std::string ANSI_COLOURS[6] = {
    "\u001B[31m", //Red
    "\u001B[32m", //Green
    "\u001B[33m", //Yellow
    "\u001B[34m", //Blue
    "\u001B[35m", //Purple
    "\u001B[36m"  //Cyan
};
const int NUM_ANSI_COLOURS = 6;