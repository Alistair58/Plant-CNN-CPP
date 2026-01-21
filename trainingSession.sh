./Plant-CNN-CPP train 75
./Plant-CNN-CPP test 1000
./Plant-CNN-CPP test 1000 ds=train
for i in $(seq 1 30);
do
    echo $i
    ./Plant-CNN-CPP train 250
    ./Plant-CNN-CPP test 1000
    ./Plant-CNN-CPP test 1000 ds=train
done
