cd ./build
cmake .. -G "MinGW Makefiles" -DDEBUG=0
mingw32-make
cd ..
build\Plant-CNN-CPP.exe
