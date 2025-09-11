Plant-CNN-CPP.exe train 350
Plant-CNN-CPP.exe test 1000
Plant-CNN-CPP.exe test 1000 ds=train

for /l %%x in (1, 1, 100) do (
    echo %%x
    Plant-CNN-CPP.exe train 500
    Plant-CNN-CPP.exe test 1000
    Plant-CNN-CPP.exe test 1000 ds=train
    
    
)
