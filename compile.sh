cd ./build
cmake ..  -DTEST=0 -DDEBUG=0 -DPROFILING=0 -DASAN=0
make
mv "Plant-CNN-CPP" ".."
cd ..