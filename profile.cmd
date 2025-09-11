cd ./build
cmake .. -G "MinGW Makefiles" -DPROFILING=1
mingw32-make
move "Plant-CNN-CPP.exe" ".."
cd ..