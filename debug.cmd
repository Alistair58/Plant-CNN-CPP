cd ./build
cmake .. -G "MinGW Makefiles" -DDEBUG=1
mingw32-make
cd ..
gdb build\Plant-CNN-CPP.exe
