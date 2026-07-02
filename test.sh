cd ./build
cmake ..  -DTEST=1 -DDEBUG=0 -DPROFILING=0 -DASAN=0 -Wno-dev
make
mv "Plant-CNN-CPP" "../Plant-CNN-CPP-Test"
cd ..
./Plant-CNN-CPP-Test