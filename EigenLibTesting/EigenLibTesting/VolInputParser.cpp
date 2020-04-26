#include "VolInputParser.h"
#pragma warning(disable : 4996)


VolInputParser::VolInputParser()
{
	DummyTensor = Eigen::Tensor<myTensorType, 3>(2, 3, 2);
	DummyTensor.setValues({ { { 1,7 },{ 3,9 },{ 5,11 } },
							{ { 2,8 },{ 4,10 },{ 6,12 } } });

	std::cout << "Dummy: " << std::endl << DummyTensor << std::endl;

	//std::cout << "Dummy: " << std::endl << DummyTensor(1,2,1) << std::endl;

	//rw.wFile = fopen("erg.txt", "w"); //Open document to write into
}

VolInputParser::VolInputParser(char * txtname)
{
	//rw.wFile = fopen("erg.txt", "w"); //Open document to write into

	readInputVol(txtname);//read in the data
}


VolInputParser::~VolInputParser()
{
	//TODO write any remaining bits
	//fclose(rw.wFile);
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
	
	for (int z = 0; z<int(*sizeZ); z++) {
		for (int y= 0; y<int(*sizeY); y++) {
			for (int x = 0; x<int(*sizeX); x++) {
				//if (hCount == 488352 + 11) std::cout << int(pData[hCount]) << " at X: " << x << " Y: " << y << " Z: " << z << std::endl;
				TensorData(x, y, z) = int(pData[hCount++]); //TODO use Map function for this
			}
		}
	}

	fclose(fp);
	delete pData;
}

void VolInputParser::writeData(unsigned char * data, int numBytes)
{
	fwrite(data, 1, numBytes, rw.wFile); //writes numBytes byte(char) into file
}


//method to safe data bitwise //TODO von vorne nach hinten also 63...0 bit codieren nicht anders herum!
void VolInputParser::writeBit(uint64_t bits, int numBits)
{
	if (numBits<=rw.numWBit) {//we have free bit, just write them in
		rw.wbyte |= bits << ((rw.numWBit + 1)-numBits);
		rw.numWBit -= numBits;
	}
	else {// write as many bit as we can, store the rest again
		if (rw.numWBit >=0) {
			rw.wbyte |= bits >> (numBits - (rw.numWBit + 1));
		}

		writeData((unsigned char *) & rw.wbyte, sizeof(rw.wbyte));

		numBits -= rw.numWBit + 1;
		rw.wbyte = 0;
		rw.wbyte |= bits << (63 - numBits);
		rw.numWBit = 63 - numBits;
	}
}

void VolInputParser::writeRemainingBit()
{
	if(rw.numWBit<63)
		writeData((unsigned char *)& rw.wbyte, sizeof(rw.wbyte));
}

//write the dimensions as short(), scale as double 
void VolInputParser::writeCharacteristicData(int dim1, int dim2, int dim3, double scale)
{
	rw.wFile = fopen("erg.txt", "w"); //Open document to write into
	/*writeData((unsigned char*)& dim1, sizeof(unsigned short));
	writeData((unsigned char*)& dim2, sizeof(unsigned short));
	writeData((unsigned char*)& dim3, sizeof(unsigned short));
	writeData((unsigned char*)& scale, sizeof(double));*/

	std::cout << sizeof(int) << std::endl;

	writeBit(uint64_t(dim1), 16);
	writeBit(uint64_t(dim2), 16);
	writeBit(uint64_t(dim3), 16);

	writeRemainingBit();

	fclose(rw.wFile);

	//approach with fstream


}
