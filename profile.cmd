cd ./build
cmake .. -G "MinGW Makefiles" -DDEBUG=0 -DPROFILING=1
mingw32-make
move "Plant-CNN-CPP.exe" ".."
cd ..