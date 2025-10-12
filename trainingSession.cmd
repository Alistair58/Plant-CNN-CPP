Plant-CNN-CPP.exe train 250 rs=true
Plant-CNN-CPP.exe test 1000
Plant-CNN-CPP.exe test 1000 ds=train
for /l %%x in (1, 1, 30) do (
    echo %%x
    Plant-CNN-CPP.exe train 250
    Plant-CNN-CPP.exe test 1000
    Plant-CNN-CPP.exe test 1000 ds=train
)
