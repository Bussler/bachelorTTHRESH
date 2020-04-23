#include "VolInputParser.h"
#pragma warning(disable : 4996)


VolInputParser::VolInputParser()
{
	DummyTensor = Eigen::Tensor<myTensorType, 3>(2, 3, 2);
	DummyTensor.setValues({ { { 1,7 },{ 3,9 },{ 5,11 } },
							{ { 2,8 },{ 4,10 },{ 6,12 } } });

	/*DummyTensor = Eigen::Tensor<myTensorType, 3>(3, 3, 3);
	int counter = 0;
	for (int i = 0;i < 3;i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				DummyTensor(j,k,i) = counter++;
			}
		}
	}*/
	std::cout << "Dummy: " << std::endl << DummyTensor << std::endl;

	//std::cout << "Dummy: " << std::endl << DummyTensor(1,2,1) << std::endl;

}

VolInputParser::VolInputParser(char * txtname)
{
	readInputVol(txtname);
}


VolInputParser::~VolInputParser()
{
}

void VolInputParser::readInputVol(char * txtname)
{
	std::cout << "StartingParse" << std::endl;

	FILE *fp = fopen(txtname, "rb"); // open in binary
	if (fp == NULL) {
		std::cout << "Error opening file";
		exit(1);
	}

	unsigned short sizeX[1];
	fread((void*)sizeX, 1, sizeof(unsigned short), fp);

	unsigned short sizeY[1];
	fread((void*)sizeY, 1, sizeof(unsigned short), fp);

	unsigned short sizeZ[1];
	fread((void*)sizeZ, 1, sizeof(unsigned short), fp);

	std::cout << std::endl << "Reference Data: x, y, z: " << int(*sizeX) << " " << int(*sizeY) << " " << int(*sizeZ) << std::endl;

	TensorData = Eigen::Tensor<myTensorType, 3>(int(*sizeX), int(*sizeY), int(*sizeZ)); //creating tensor of the read in dimensions

	int uCount = int(sizeX[0])*int(sizeY[0])*int(sizeZ[0]);
	unsigned short *pData = new unsigned short[uCount];
	fread((void*)pData,uCount, sizeof(unsigned short), fp);
	
	/*for (int i = 488352; i < 488352 +16; i++) { //488352
		std::cout << "Parse1: " << int(pData[i]) << std::endl;
	}*/
	
	//read in the array into Tensor slice by slice TODO: instantly read in the file? Maybe use iffile for read
	int hCount = 0;

	//TODO: wir hätten in Matrixform hier schon das 1 Mode unfolding! -> nicht in Tensor, sondern in Matrix einlesen!
	for (int z = 0; z<int(*sizeZ); z++) {
		for (int y= 0; y<int(*sizeY); y++) {
			for (int x = 0; x<int(*sizeX); x++) {
				//if (hCount == 488352 + 11) std::cout << int(pData[hCount]) << " at X: " << x << " Y: " << y << " Z: " << z << std::endl;
				TensorData(x, y, z) = int(pData[hCount++]);
			}
		}
	}

	fclose(fp);
	delete pData;
}