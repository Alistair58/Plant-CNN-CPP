#include "cnnutils.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>  

class MockCnnUtils: public CnnUtils {
    public:
        MockCnnUtils(
            std::vector<int> numNeurons,
            std::vector<dimens> mapDimens,
            std::vector<std::pair<int,int>> kernelSizes,
            std::vector<std::pair<int,int>> strides,
            bool padding,
            std::vector<Tensor> kernels,
            std::vector<Tensor> weights      
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
    std::vector<Tensor> weights 
    ){
    this->numNeurons = numNeurons;
    this->mapDimens = mapDimens;
    this->kernelSizes = kernelSizes;
    this->strides = strides;
    this->padding = padding;
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
    MockCnnUtils *cnnUtils;
    void SetUp() override {
        std::vector<int> numNeurons = {1920,1920,47};
        //includes the result of pooling (except final pooling)
        std::vector<dimens> mapDimens(4);
        mapDimens[0] = {3,128,128};
        mapDimens[1] = {30,64,64};
        mapDimens[2] = {60,32,32};
        mapDimens[3] = {120,16,16};
        std::vector<std::pair<int,int>> kernelSizes(3);
        kernelSizes[0] = {3,3};
        kernelSizes[1] = {3,3};
        kernelSizes[2] = {3,3};
        std::vector<std::pair<int,int>> strides(4);
        strides[0] = {2,2};
        strides[1] = {2,2};
        strides[2] = {2,2};
        strides[3] = {4,4};
        std::vector<Tensor> kernels; //TODO
        std::vector<Tensor> weights;
        cnnUtils = new MockCnnUtils(numNeurons,mapDimens,kernelSizes,strides,true,kernels,weights);
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
    std::vector<float> largeCase3Correct = {1.00000000e+00,7.17509597e-66,3.13913279e-17,3.22134029e-27};
    std::vector<float> largeCase3Res = CnnUtils::softmax(largeCase3);
    for(int i{0};i<largeCase3.size();i++){
        EXPECT_FLOAT_EQ(largeCase3Res[i],largeCase3Correct[i]);
    } 
}


TEST_F(CnnUtilsTest,dotProduct8fWorks){

}

TEST_F(CnnUtilsTest,horizontalSumWorks){

}

TEST_F(CnnUtilsTest,floorModWorks){   
    
}

TEST_F(CnnUtilsTest,applyGradientsWorks){

}

TEST_F(CnnUtilsTest,resetGradWorks){

}

TEST_F(CnnUtilsTest,fixedSizedConvolutionWorks){

}

TEST_F(CnnUtilsTest,prePaddedConvolutionWorks){

}

TEST_F(CnnUtilsTest,variableSizedConvolutionWorks){

}

TEST_F(CnnUtilsTest,maxPoolWorks){

}

TEST_F(CnnUtilsTest,gaussianBlurKernelWorks){

}

TEST_F(CnnUtilsTest,normaliseImgWorks){

}

TEST_F(CnnUtilsTest,parseImgWorks){

}