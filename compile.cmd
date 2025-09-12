cd ./build
cmake .. -G "MinGW Makefiles" -DDEBUG=0 -DPROFILING=0
mingw32-make
move "Plant-CNN-CPP.exe" ".."
cd ..