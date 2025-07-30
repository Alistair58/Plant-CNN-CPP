cd ./build
cmake .. -G "MinGW Makefiles"
mingw32-make
cd ..
gdb build\Plant-CNN-CPP.exe
