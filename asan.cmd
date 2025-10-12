cmake -S . -B build -G "Ninja" -DASAN=1 -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang++.exe"
cmake --build build
move "build\Plant-CNN-CPP.exe" .

