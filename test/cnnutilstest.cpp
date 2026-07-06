#include "cnnutils.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>  

class MockCnnUtils: public CnnUtils {
    public:
        //Make some of the attributes public for easier testing
        using CnnUtils::kernels;
        using CnnUtils::weights;
        using CnnUtils::maps;
        using CnnUtils::activations;
        using CnnUtils::kernelsGrad;
        using CnnUtils::weightsGrad;
        MockCnnUtils(
            std::vector<int> numNeurons,
            std::vector<dimens> mapDimens,
            std::vector<std::pair<int,int>> kernelSizes,
            std::vector<std::pair<int,int>> strides,
            bool padding,
            std::vector<Tensor> kernels,
            std::vector<Tensor> weights,
            float LR      
        );
        MOCK_METHOD(std::vector<Tensor>, loadWeights, (bool loadNew), (override));
        MOCK_METHOD(std::vector<Tensor>, loadKernels, (bool loadNew), (override));
        MOCK_METHOD(void, saveWeights, (), (override));
        MOCK_METHOD(void, saveKernels, (), (override));
        MOCK_METHOD(void, resetWeights, (), (override));
        MOCK_METHOD(void, resetKernels, (), (override));
};

MockCnnUtils::MockCnnUtils(
    std::vector<int> numNeurons,
    std::vector<dimens> mapDimens,
    std::vector<std::pair<int,int>> kernelSizes,
    std::vector<std::pair<int,int>> strides,
    bool padding,
    std::vector<Tensor> kernels,
    std::vector<Tensor> weights,
    float LR
    ){
    this->numNeurons = numNeurons;
    this->mapDimens = mapDimens;
    this->kernelSizes = kernelSizes;
    this->strides = strides;
    this->padding = padding;
    this->kernels = kernels;
    this->weights = weights;
    this->LR = LR;
    //Same code as CNN constructor
    kernelsGrad = std::vector<Tensor>(kernels.size());
    for(int i=0;i<kernels.size();i++){
        kernelsGrad[i] = Tensor(kernels[i].getDimens()); 
        Tensor biasesGrad(kernels[i].getBiases()->getDimens());
        kernelsGrad[i].setBiases(biasesGrad);
    }
    weightsGrad = std::vector<Tensor>(weights.size());
    for(int i=0;i<weights.size();i++){
        weightsGrad[i] = Tensor(weights[i].getDimens());
        Tensor biasesGrad(weights[i].getBiases()->getDimens());
        weightsGrad[i].setBiases(biasesGrad);
    }
    this->activations = std::vector<Tensor>(numNeurons.size());
    for(int l=0;l<numNeurons.size();l++){
        activations[l] = Tensor({numNeurons[l]});
    }
    this->maps = std::vector<Tensor>(mapDimens.size());
    for(int l=0;l<mapDimens.size();l++){
        maps[l] = Tensor({mapDimens[l].c,mapDimens[l].h,mapDimens[l].w});
    }
    if(padding){
        this->paddedMaps = std::vector<Tensor>(mapDimens.size()-1); //last map is pooled not convolved - my favourite way of having a Martini
        for(int l=0;l<mapDimens.size()-1;l++){
            int kernelRadiusY = std::floor(this->kernelSizes[l].first/2);
            int kernelRadiusX = std::floor(this->kernelSizes[l].second/2);
            int paddedHeight = mapDimens[l].h+2*kernelRadiusY;
            int paddedWidth = mapDimens[l].w+2*kernelRadiusX;
            paddedMaps[l] = Tensor({mapDimens[l].c,paddedHeight,paddedWidth});
        }
    }
    for(int l=0;l<kernelSizes.size();l++){
        if(kernelSizes[l].first==0 || kernelSizes[l].second==0){ //pooling
            int pooledDimenX = mapDimens[l].w/strides[l].second;
            int pooledDimenY = mapDimens[l].h/strides[l].first;
            maxPoolIndices.push_back(std::unique_ptr<int[]>(new int[mapDimens[l].c*pooledDimenY*pooledDimenX]));
        }
    }
    //final pooling
    int finalPooledDimenX = mapDimens[mapDimens.size()-1].w/strides[strides.size()-1].second;
    int finalPooledDimenY = mapDimens[mapDimens.size()-1].h/strides[strides.size()-1].first;
    maxPoolIndices.push_back(std::unique_ptr<int[]>(
        new int[mapDimens[mapDimens.size()-1].c*finalPooledDimenY*finalPooledDimenX]
    ));
};


class CnnUtilsTest : public testing::Test{
    protected:
        MockCnnUtils *cnnUtils;
        void SetUp() override {
            std::vector<int> numNeurons = {4,2};
            //includes the result of pooling (except final pooling)
            std::vector<dimens> mapDimens(2);
            mapDimens[0] = {2,3,3};
            mapDimens[1] = {2,2,2};
            std::vector<std::pair<int,int>> kernelSizes(1);
            kernelSizes[0] = {3,3};
            std::vector<std::pair<int,int>> strides(2);
            strides[0] = {2,2};
            strides[1] = {2,2};
            float LR = 0.1;

            std::vector<Tensor> kernels(1);
            kernels[0] = Tensor({2,2,3,3});
            kernels[0].slice({0,0}) = {1, -1,  2,
                                       2,  1, -3,
                                       0,  3, -1};
            kernels[0].slice({0,1}) = {0,  2,   2,
                                       1,  0.5, 0.2,
                                       2,  1,  -1.5};
            kernels[0].slice({1,0}) = {0.1, 0.5, -0.5,
                                       1,  -1,    1,
                                       2,  -0.5,  2};
            kernels[0].slice({1,1}) = {-1,  2,    0,
                                       0.2, 1.5, -1,
                                       1,   0.5,  0.2};
            Tensor kernelBiases({2});
            kernelBiases = {1,-1};
            kernels[0].setBiases(kernelBiases);

            std::vector<Tensor> weights(1);
            weights[0] = Tensor({2,4});
            weights[0].slice({0}) = {1,   2, -3, 4};
            weights[0].slice({1}) = {0.1, 5, -1, 3.5};
            Tensor weightBiases({2});
            weightBiases = {0.1,-0.1};
            weights[0].setBiases(weightBiases);
            
            cnnUtils = new MockCnnUtils(numNeurons,mapDimens,kernelSizes,strides,true,kernels,weights,LR);
        }

        void TearDown() override {
            delete cnnUtils;
        }
};

TEST_F(CnnUtilsTest,reluWorks){   
    EXPECT_FLOAT_EQ(CnnUtils::relu(7),7);
    EXPECT_FLOAT_EQ(CnnUtils::relu(0),0);
    EXPECT_FLOAT_EQ(CnnUtils::relu(-1),0);
    EXPECT_FLOAT_EQ(CnnUtils::relu(-100000000000),0);
    EXPECT_FLOAT_EQ(CnnUtils::relu(123456789),123456789);
}

TEST_F(CnnUtilsTest,leakyReluWorks){   
    EXPECT_FLOAT_EQ(CnnUtils::leakyRelu(7),7);
    EXPECT_FLOAT_EQ(CnnUtils::leakyRelu(0),0);
    EXPECT_FLOAT_EQ(CnnUtils::leakyRelu(-1),-0.01);
    EXPECT_FLOAT_EQ(CnnUtils::leakyRelu(-100000000000),-1000000000);
    EXPECT_FLOAT_EQ(CnnUtils::leakyRelu(123456789),123456789);
}

TEST_F(CnnUtilsTest,floatCmpWorks){   
    EXPECT_TRUE(CnnUtils::floatCmp(7.0f,7.1f,0.1f));
    EXPECT_TRUE(CnnUtils::floatCmp(7.0f,7.001f,0.001f));
    EXPECT_TRUE(CnnUtils::floatCmp(0.29999999999999999999999f,0.3f));
    EXPECT_TRUE(CnnUtils::floatCmp(7.0f,7.0f,0.1f));
    EXPECT_TRUE(CnnUtils::floatCmp(7.0f,7.0f,0.001f));
    EXPECT_TRUE(CnnUtils::floatCmp(0.1f,-0.1f,0.5f));
    EXPECT_TRUE(CnnUtils::floatCmp(0.1f,-0.1f,0.2f));

    EXPECT_FALSE(CnnUtils::floatCmp(0.1f,-0.1f,0.0f));
    EXPECT_FALSE(CnnUtils::floatCmp(7.0f,7.1f,0.01f));
    EXPECT_FALSE(CnnUtils::floatCmp(7.0f,7.1f,0.01f));
}

TEST_F(CnnUtilsTest,sigmoidWorks){   
    EXPECT_FLOAT_EQ(CnnUtils::sigmoid(0),0.5f);
    EXPECT_FLOAT_EQ(CnnUtils::sigmoid(1),0.73105857863f);
    EXPECT_FLOAT_EQ(CnnUtils::sigmoid(-1),0.26894142137f);
    EXPECT_FLOAT_EQ(CnnUtils::sigmoid(-20),2.0611536182e-9f);
    EXPECT_FLOAT_EQ(CnnUtils::sigmoid(20),0.999999997939f);
    EXPECT_FLOAT_EQ(CnnUtils::sigmoid(0.001f),0.500249999979f);
}

TEST_F(CnnUtilsTest,softmaxWorks){   
    std::vector<float> typicalCase1 = {1,2,3,4,5};
    //From scipy 
    std::vector<float> typicalCase1Correct = {0.01165623f,0.03168492f,0.08612854f,0.23412166f,0.63640865f};
    std::vector<float> typicalCase1Res = CnnUtils::softmax(typicalCase1);
    for(int i{0};i<typicalCase1.size();i++){
        EXPECT_FLOAT_EQ(typicalCase1Res[i],typicalCase1Correct[i]);
    }

    std::vector<float> typicalCase2 = {7,1,-5,2,0.2};
    std::vector<float> typicalCase2Correct = {9.89769134e-01,2.45339240e-03,6.08135174e-06,6.66901197e-03,1.10238026e-03};
    std::vector<float> typicalCase2Res = CnnUtils::softmax(typicalCase2);
    for(int i{0};i<typicalCase2.size();i++){
        EXPECT_FLOAT_EQ(typicalCase2Res[i],typicalCase2Correct[i]);
    }
    
    std::vector<float> largeCase1 = {91,100,0,56};
    std::vector<float> largeCase1Correct = {1.23394576e-04,9.99876605e-01,3.71961694e-44,7.78017209e-20};
    std::vector<float> largeCase1Res = CnnUtils::softmax(largeCase1);
    for(int i{0};i<largeCase1.size();i++){
        EXPECT_FLOAT_EQ(largeCase1Res[i],largeCase1Correct[i]);
    }

    std::vector<float> largeCase2 = {-100,-5,-61,-67};
    std::vector<float> largeCase2Correct = {5.52108228e-42,1.00000000e+00,4.78089288e-25,1.18506486e-27};
    std::vector<float> largeCase2Res = CnnUtils::softmax(largeCase2);
    for(int i{0};i<largeCase2.size();i++){
        EXPECT_FLOAT_EQ(largeCase2Res[i],largeCase2Correct[i]);
    } 

    std::vector<float> largeCase3 = {50,-100,12,-11};
    std::vector<float> largeCase3Correct = {1.00000000e+00,0,3.13913279e-17,3.22134029e-27}; //7.17509597e-66
    std::vector<float> largeCase3Res = CnnUtils::softmax(largeCase3);
    for(int i{0};i<largeCase3.size();i++){
        EXPECT_FLOAT_EQ(largeCase3Res[i],largeCase3Correct[i]);
    } 
}


TEST_F(CnnUtilsTest,dotProduct8fWorks){
    float xTypicalCase1[] = {1, 2, 3,4,5,6,7,8};
    float yTypicalCase1[] = {11,30,1,0,2,1,2,91};
    float typicalCase1Correct = 832.0f;
    float typicalCase1Res = CnnUtils::dotProduct8f(xTypicalCase1,yTypicalCase1);
    EXPECT_FLOAT_EQ(typicalCase1Res,typicalCase1Correct);

    float xTypicalCase2[] = { 1.5f,-2.7f,   300,42.1,-67.1,12,   0.01f, -0.22f};
    float yTypicalCase2[] = {-1,   12.123f,-7,  202,  3,  -1.1f, 0.01f, -101};
    float typicalCase2Correct = 6177.688;
    float typicalCase2Res = CnnUtils::dotProduct8f(xTypicalCase2,yTypicalCase2);
    EXPECT_FLOAT_EQ(typicalCase2Res,typicalCase2Correct);

    float xSamePtr[] = {-60,1,2,3,44,0.1,-0.1,75.75f};
    float samePtrCorrect = 11288.0825f;
    float samePtrRes = CnnUtils::dotProduct8f(xSamePtr,xSamePtr);
    EXPECT_FLOAT_EQ(samePtrRes,samePtrCorrect);
}

TEST_F(CnnUtilsTest,horizontalSumWorks){
    float typicalCase1Arr[] = {1,2,3,4,5,6,7,8};
    __m256 typicalCase1 = _mm256_loadu_ps(typicalCase1Arr);
    float typicalCase1Correct = 36;
    float typicalCase1Res =  CnnUtils::horizontalSum(typicalCase1);
    EXPECT_FLOAT_EQ(typicalCase1Res,typicalCase1Correct);

    float typicalCase2Arr[] = {-1, 2.2, 300.34f, -6767.1f, 23, -0.01f, 0, 1};
    __m256 typicalCase2 = _mm256_loadu_ps(typicalCase2Arr);
    float typicalCase2Correct = -6441.57f;
    float typicalCase2Res =  CnnUtils::horizontalSum(typicalCase2);
    EXPECT_FLOAT_EQ(typicalCase2Res,typicalCase2Correct);
}

TEST_F(CnnUtilsTest,floorModWorks){   
    int xTypicalCase1 = -5;
    int yTypicalCase1 = 2;
    int typicalCase1Correct = 1;
    int typicalCase1Res = CnnUtils::floorMod(xTypicalCase1,yTypicalCase1);
    EXPECT_FLOAT_EQ(typicalCase1Res,typicalCase1Correct);

    int xTypicalCase2 = 7;
    int yTypicalCase2 = -3;
    int typicalCase2Correct = -1;
    int typicalCase2Res = CnnUtils::floorMod(xTypicalCase2,yTypicalCase2);
    EXPECT_FLOAT_EQ(typicalCase2Res,typicalCase2Correct);

    int xTypicalCase3 = -6;
    int yTypicalCase3 = -7;
    int typicalCase3Correct = -6;
    int typicalCase3Res = CnnUtils::floorMod(xTypicalCase3,yTypicalCase3);
    EXPECT_FLOAT_EQ(typicalCase3Res,typicalCase3Correct);

    int xLargeCase1 = 12345;
    int yLargeCase1 = -6789;
    int largeCase1Correct = -5556;
    int largeCase1Res = CnnUtils::floorMod(xLargeCase1,yLargeCase1);
    EXPECT_FLOAT_EQ(largeCase1Res,largeCase1Correct);
}

TEST_F(CnnUtilsTest,applyGradientsSimpleCase){
    //2x2x3x3
    cnnUtils->kernelsGrad[0].slice({0,0}) = {1,  1,  1,
                                1,  1,  1,
                                1,  1,  1};
    cnnUtils->kernelsGrad[0].slice({0,1}) = {1,  1,  1,
                                1,  1,  1,
                                1,  1,  1};
    cnnUtils->kernelsGrad[0].slice({1,0}) = {1,  1,  1,
                                1,  1,  1,
                                1,  1,  1};
    cnnUtils->kernelsGrad[0].slice({1,1}) = {1,  1,  1,
                                1,  1,  1,
                                1,  1,  1};
    Tensor kernelsBiasesGrad({2});
    kernelsBiasesGrad = {1,1};
    cnnUtils->kernelsGrad[0].setBiases(kernelsBiasesGrad);

    //2x4
    cnnUtils->weightsGrad[0].slice({0}) = {1, 1, 1, 1};
    cnnUtils->weightsGrad[0].slice({1}) = {1, 1, 1, 1};
    Tensor weightsBiasesGrad({2});
    weightsBiasesGrad = {1,1};
    cnnUtils->weightsGrad[0].setBiases(weightsBiasesGrad);


    //LR is 0.1 and so each value should be 0.1 lower
    std::vector<Tensor> kernelsCorrect(1);
    kernelsCorrect[0] = Tensor({2,2,3,3});
    kernelsCorrect[0].slice({0,0}) = {0.9, -1.1,  1.9,
                                1.9,  0.9, -3.1,
                                -0.1,  2.9, -1.1};
    kernelsCorrect[0].slice({0,1}) = {-0.1,  1.9,   1.9,
                                0.9,  0.4, 0.1,
                                1.9,  0.9,  -1.6};
    kernelsCorrect[0].slice({1,0}) = {0, 0.4, -0.6,
                                0.9,  -1.1,    0.9,
                                1.9,  -0.6,  1.9};
    kernelsCorrect[0].slice({1,1}) = {-1.1,  1.9, -0.1,
                                0.1, 1.4, -1.1,
                                0.9, 0.4,  0.1};
    Tensor kernelsBiasesCorrect({2});
    kernelsBiasesCorrect = {0.9,-1.1};
    

    std::vector<Tensor> weightsCorrect(1);
    weightsCorrect[0] = Tensor({2,4});
    weightsCorrect[0].slice({0}) = {0.9,   1.9, -3.1, 3.9};
    weightsCorrect[0].slice({1}) = {0, 4.9, -1.1, 3.4};
    Tensor weightsBiasesCorrect({2});
    weightsBiasesCorrect = {0,-0.2};


    const int batchSize = 1;
    cnnUtils->applyGradients(batchSize);

    for(int i=0;i<cnnUtils->kernels[0].getTotalSize();i++){
        EXPECT_FLOAT_EQ(*(cnnUtils->kernels[0])[i],*(kernelsCorrect[0])[i]);
    }
    for(int i=0;i<cnnUtils->kernels[0].getBiases()->getTotalSize();i++){
        EXPECT_FLOAT_EQ(*(*cnnUtils->kernels[0].getBiases())[i],*kernelsBiasesCorrect[i]);
    }

    for(int i=0;i<cnnUtils->weights[0].getTotalSize();i++){
        EXPECT_FLOAT_EQ(*(cnnUtils->weights[0])[i],*(weightsCorrect[0])[i]);
    }
    for(int i=0;i<cnnUtils->weights[0].getBiases()->getTotalSize();i++){
        EXPECT_FLOAT_EQ(*(*cnnUtils->weights[0].getBiases())[i],*weightsBiasesCorrect[i]);
    }
}

TEST_F(CnnUtilsTest,applyGradientsTypicalCase){
    //2x2x3x3
    cnnUtils->kernelsGrad[0].slice({0,0}) = {0.4,  -0.4,  2,
                                -2,  -4,  1.6,
                                4,  -2.4,  2};
    cnnUtils->kernelsGrad[0].slice({0,1}) = {0.4,  -0.4,  0.4,
                                20,  0,  -4,
                                -2.4,  -1.2,  0.8};
    cnnUtils->kernelsGrad[0].slice({1,0}) = {2.8,  0.4,  0,
                                0,  0.8,  0,
                                2.4,  -26.8,  0.4};
    cnnUtils->kernelsGrad[0].slice({1,1}) = {4.4,  -8.8,  0,
                                2,  2.4,  2.8,
                                -4,  -8,  -12};
    Tensor kernelsBiasesGrad({2});
    kernelsBiasesGrad = {2,-8.4};
    cnnUtils->kernelsGrad[0].setBiases(kernelsBiasesGrad);

    //2x4
    cnnUtils->weightsGrad[0].slice({0}) = {0.4, -0.8, -0.4, 0.8};
    cnnUtils->weightsGrad[0].slice({1}) = {2,0.8,-0.4,4};
    Tensor weightsBiasesGrad({2});
    weightsBiasesGrad = {-4,0};
    cnnUtils->weightsGrad[0].setBiases(weightsBiasesGrad);


    //LR is 0.1
    std::vector<Tensor> kernelsCorrect(1);
    kernelsCorrect[0] = Tensor({2,2,3,3});
    kernelsCorrect[0].slice({0,0}) = {0.99, -0.99,  1.95,
                                       2.05,  1.1, -3.04,
                                       -0.1,  3.06, -1.05};
    kernelsCorrect[0].slice({0,1}) = {-0.01,  2.01,   1.99,
                                       0.5,  0.5, 0.3,
                                       2.06,  1.03,  -1.52};
    kernelsCorrect[0].slice({1,0}) = {0.03, 0.49, -0.5,
                                       1,  -1.02,    1,
                                       1.94,  0.17,  1.99};
    kernelsCorrect[0].slice({1,1}) = {-1.11,  2.22,    0,
                                       0.15, 1.44, -1.07,
                                       1.1,   0.7,  0.5};
    Tensor kernelsBiasesCorrect({2});
    kernelsBiasesCorrect = {0.95,-0.79};
    

    std::vector<Tensor> weightsCorrect(1);
    weightsCorrect[0] = Tensor({2,4});
    weightsCorrect[0].slice({0}) = {0.99,2.02,-2.99,3.98};
    weightsCorrect[0].slice({1}) = {0.05,4.98, -0.99, 3.4};
    Tensor weightsBiasesCorrect({2});
    weightsBiasesCorrect = {0.2,-0.1};


    const int batchSize = 4;
    cnnUtils->applyGradients(batchSize);

    //Check that the kernels and weights have been modified correctly
    for(int i=0;i<cnnUtils->kernels[0].getTotalSize();i++){
        EXPECT_FLOAT_EQ(*(cnnUtils->kernels[0])[i],*(kernelsCorrect[0])[i]);
    }
    for(int i=0;i<cnnUtils->kernels[0].getBiases()->getTotalSize();i++){
        EXPECT_FLOAT_EQ(*(*cnnUtils->kernels[0].getBiases())[i],*kernelsBiasesCorrect[i]);
    }

    for(int i=0;i<cnnUtils->weights[0].getTotalSize();i++){
        EXPECT_FLOAT_EQ(*(cnnUtils->weights[0])[i],*(weightsCorrect[0])[i]);
    }
    for(int i=0;i<cnnUtils->weights[0].getBiases()->getTotalSize();i++){
        EXPECT_FLOAT_EQ(*(*cnnUtils->weights[0].getBiases())[i],*weightsBiasesCorrect[i]);
    }

    //Check that the gradients have been reset to 0
    for(int i=0;i<cnnUtils->kernelsGrad[0].getTotalSize();i++){
        EXPECT_FLOAT_EQ(*(cnnUtils->kernelsGrad[0])[i],0);
    }
    for(int i=0;i<cnnUtils->kernelsGrad[0].getBiases()->getTotalSize();i++){
        EXPECT_FLOAT_EQ(*(*cnnUtils->kernelsGrad[0].getBiases())[i],0);
    }

    for(int i=0;i<cnnUtils->weightsGrad[0].getTotalSize();i++){
        EXPECT_FLOAT_EQ(*(cnnUtils->weightsGrad[0])[i],0);
    }
    for(int i=0;i<cnnUtils->weightsGrad[0].getBiases()->getTotalSize();i++){
        EXPECT_FLOAT_EQ(*(*cnnUtils->weightsGrad[0].getBiases())[i],0);
    }
}

TEST_F(CnnUtilsTest,variableSizedConvolution3x3SimpleNoPad){
    Tensor image({3,4,4});
    image.slice({0}) = {1,1,1,1,
                        1,1,1,1,
                        1,1,1,1,
                        1,1,1,1};
    image.slice({1}) = {2,2,2,2,
                        2,2,2,2,
                        2,2,2,2,
                        2,2,2,2};
    image.slice({2}) = {4,4,4,4,
                        4,4,4,4,
                        4,4,4,4,
                        4,4,4,4};
    Tensor kernel({3,3,3});
    kernel.slice({0}) = {2,2,2,
                         2,2,2,
                         2,2,2};
    kernel.slice({1}) = {1,1,1,
                         1,1,1,
                         1,1,1};
    kernel.slice({2}) = {0,0,0,
                         0,0,0,
                         0,0,0};

    Tensor res = CnnUtils::convolution(image,kernel,1,1,false);
    
    std::vector<int> correctDimens = {2,2};
    Tensor correctRes(correctDimens);
    correctRes = {36,36,36,36};


    std::vector<int> resDimens = res.getDimens();
    ASSERT_EQ(resDimens.size(),correctDimens.size());
    for(int i=0;i<resDimens.size();i++){
        ASSERT_EQ(resDimens[i],correctDimens[i]);
    }
    for(int i=0;i<res.getTotalSize();i++){
        ASSERT_FLOAT_EQ(*res[i],*correctRes[i]);
    }
}

TEST_F(CnnUtilsTest,variableSizedConvolution3x3NoPad){
    Tensor image({3,4,4});
    image.slice({0}) = {1,1,1,1,
                        1,1,1,1,
                        1,1,0,0,
                        1,1,0,0};
    image.slice({1}) = {2,2,2,2,
                        2,-1,2,2,
                        2,-1,2,2,
                        2,2,2,2};
    image.slice({2}) = {0,4,0,4,
                        4,4,4,4,
                        4,4,0,4,
                        4,4,4,4};
    Tensor kernel({3,3,3});
    kernel.slice({0}) = {1,2,3,
                         1,1,1,
                         3,2,1};
    kernel.slice({1}) = {0,1,0,
                         0,1,0,
                         0,1,0};
    kernel.slice({2}) = {-1,0,0,
                         1,-5,0,
                         0,0,0};

    Tensor res = CnnUtils::convolution(image,kernel,1,1,false);
    
    std::vector<int> correctDimens = {2,2};
    Tensor correctRes(correctDimens);
    correctRes = {-0.02,-0.02,-0.07,16};


    std::vector<int> resDimens = res.getDimens();
    ASSERT_EQ(resDimens.size(),correctDimens.size());
    for(int i=0;i<resDimens.size();i++){
        ASSERT_EQ(resDimens[i],correctDimens[i]);
    }
    for(int i=0;i<res.getTotalSize();i++){
        ASSERT_FLOAT_EQ(*res[i],*correctRes[i]);
    }
}



TEST_F(CnnUtilsTest,variableSizedConvolution5x5Stride2NoPad){

}

//3x3 is treated specially
//x stride 1 on 3x3 is also treated differently
TEST_F(CnnUtilsTest,variableSizedConvolution3x3Stride1NoPad){

}

TEST_F(CnnUtilsTest,variableSizedConvolution3x3Stride2NoPad){

}

//Anything with width >= 8 is treated specially
TEST_F(CnnUtilsTest,variableSizedConvolution9x9Stride2NoPad){

}


TEST_F(CnnUtilsTest,fixedSizedConvolutionWorks){

}

TEST_F(CnnUtilsTest,prePaddedConvolutionWorks){

}



TEST_F(CnnUtilsTest,maxPoolWorks){

}

TEST_F(CnnUtilsTest,gaussianBlurKernelWorks){

}

TEST_F(CnnUtilsTest,normaliseImgWorks){

}

TEST_F(CnnUtilsTest,parseImgWorks){

}