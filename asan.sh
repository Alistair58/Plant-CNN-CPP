cd ./build
export ASAN_OPTIONS="
detect_leaks=1
abort_on_error=1
strict_string_checks=1
detect_stack_use_after_return=1
check_initialization_order=1
alloc_dealloc_mismatch=1
"
cmake ..  -DASAN=1
make
mv "Plant-CNN-CPP" ".."
cd ..