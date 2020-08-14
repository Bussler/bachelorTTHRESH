#include "VolInputParser.h"
#pragma warning(disable : 4996)

struct BitIO::RWWrapper {

	uint64_t wbyte = 0; //store for 8 byte
	uint64_t rbyte = 0;

	int numWBit = 63; //indicates free bit
	int numRBit = -1;

	int vergWBit = 0;
	int readRBit = 64;

	FILE * wFile;
	FILE * rFile;

}BitIO::rw;

void BitIO::writeData(unsigned char * data, int numBytes)
{
	fwrite(data, 1, numBytes, rw.wFile); //writes numBytes byte(char) into file
}

void BitIO::writeBit(uint64_t bits, int numBits)
{

	if (numBits + rw.vergWBit <= 64) {//we have free bit, just write them in
		rw.wbyte |= bits << (rw.vergWBit);
		rw.vergWBit += numBits;
	}
	else {// write as many bit as we can, store the rest again

		if (rw.vergWBit < 64) {//if there is still something free, squeeze it in
			rw.wbyte |= (bits << (rw.vergWBit));
		}

		writeData((unsigned char *)& rw.wbyte, sizeof(rw.wbyte));

		numBits -= 64 - rw.vergWBit;
		rw.wbyte = 0;
		rw.wbyte |= (bits >> (64 - rw.vergWBit));
		rw.vergWBit = 0 + numBits;
	}
}

void BitIO::writeRemainingBit()
{
	if (rw.vergWBit > 0) {
		writeData((unsigned char *)& rw.wbyte, sizeof(rw.wbyte));
	}

}

void BitIO::readData(uint8_t * buf, int numBytes)
{
	fread(buf, 1, numBytes, rw.rFile);
}

uint64_t BitIO::readBit(int numBits)
{
	uint64_t result = 0;
	if (numBits + rw.readRBit <= 64) {//we haven't read everything from the buffer
		int amtShift = 64 - numBits - rw.readRBit;
		result |= rw.rbyte << amtShift >> (amtShift + rw.readRBit);
		rw.readRBit += numBits;
	}
	else {
		if (rw.readRBit < 64) { //read as much as possible
			result |= rw.rbyte >> rw.readRBit;
		}

		readData((uint8_t *)& rw.rbyte, sizeof(rw.rbyte));

		numBits -= 64 - rw.readRBit;

		int amtShift = 64 - numBits - 0;
		result |= rw.rbyte << amtShift >> (amtShift + 0) << (64 - rw.readRBit);//shift to the amount of already read in data

		rw.readRBit = 0 + numBits;
	}

	return result;
}

void BitIO::openRead(char * name)
{
	rw.rFile = fopen(name, "rb");
}

void BitIO::closeRead()
{
	fclose(rw.rFile);
	rw.rbyte = 0;
	rw.readRBit = 64;
}

void BitIO::openWrite(char * name)
{
	rw.wFile = fopen(name, "wb");
}

void BitIO::closeWrite()
{
	fclose(rw.wFile);
	rw.wbyte = 0;
	rw.vergWBit = 0;
}



VolInputParser::VolInputParser()
{
	/*DummyTensor = Eigen::Tensor<myTensorType, 3>(2, 3, 2);
	DummyTensor.setValues({ { { 1,7 },{ 3,9 },{ 5,11 } },
							{ { 2,8 },{ 4,10 },{ 6,12 } } });*/

	int y = 7;//4; 7; 277
	int x = 5;//3; 5; 277
	int z = 2;//2; 164
	//DummyTensor = Eigen::Tensor<myTensorType, 3>(y,x,z);
	TensorData = Eigen::Tensor<myTensorType, 3>(y, x, z);
	double counter = 0;
	for (int i = 0;i < y; i++) {
		for (int j = 0;j < x; j++) {
			for (int k = 0;k < z;k++) {
				//DummyTensor(i,j,k) = counter++;
				TensorData(i, j, k) = counter++;
			}
		}
	}

	std::cout << "Dummy: " << std::endl << TensorData << std::endl;
}

VolInputParser::VolInputParser(char * txtname)
{
	readInputVol(txtname);//read in the data
	//readRawInputVol(txtname);
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
	
	//read in the array into Tensor slice by slice TODO: instantly read in the file? Maybe use iffile for read
	int hCount = 0;
	
	for (int z = 0; z<int(*sizeZ); z++) {
		for (int y= 0; y<int(*sizeY); y++) {
			for (int x = 0; x<int(*sizeX); x++) {
				TensorData(x, y, z) = int(pData[hCount++]); //TODO use Map function for this
			}
		}
	}

	fclose(fp);
	delete pData;
	std::cout << "Parsing success" << std::endl;
}

void VolInputParser::readRawInputVol(char * txtname)
{
	std::cout << "Starting Parse Raw" << std::endl;

	FILE *fp = fopen(txtname, "rb"); // open in binary
	if (fp == NULL) {
		std::cout << "Error opening file";
		exit(1);
	}

	unsigned short sizeX = 512;//256;

	unsigned short sizeY = 512;//256;

	unsigned short sizeZ = 361;//256;

	std::cout << std::endl << "Reference Data: x, y, z: " << int(sizeX) << " " << int(sizeY) << " " << int(sizeZ) << std::endl;

	TensorData = Eigen::Tensor<myTensorType, 3>(int(sizeX), int(sizeY), int(sizeZ)); //creating tensor of the read in dimensions

	int uCount = int(sizeX)*int(sizeY)*int(sizeZ);
	unsigned char *pData = new unsigned char[uCount];
	fread((void*)pData, uCount, sizeof(unsigned char), fp);

	//read in the array into Tensor slice by slice TODO: instantly read in the file? Maybe use iffile for read
	int hCount = 0;

	for (int z = 0; z<int(sizeZ); z++) {
		for (int y = 0; y<int(sizeY); y++) {
			for (int x = 0; x<int(sizeX); x++) {
				TensorData(x, y, z) = int(pData[hCount++]); //TODO use Map function for this
			}
		}
	}

	fclose(fp);
	delete pData;
}

/*//method to safe data bitwise 
void VolInputParser::writeBit2(uint64_t bits, int numBits)
{
	if (numBits<=rw.numWBit+1) {//we have free bit, just write them in
		rw.wbyte |= (bits << ((rw.numWBit + 1)-numBits));
		rw.numWBit -= numBits;
	}
	else {// write as many bit as we can, store the rest again
		if (rw.numWBit >=0) {
			rw.wbyte |= (bits >> (numBits - (rw.numWBit + 1)));
		}

		//unsigned __int64 swapped= _byteswap_uint64(rw.wbyte);
		//writeData((unsigned char *)& swapped, sizeof(rw.wbyte));
		writeData((unsigned char *) & rw.wbyte, sizeof(rw.wbyte));

		numBits -= rw.numWBit + 1;
		rw.wbyte = 0;
		rw.wbyte |= (bits << (64 - numBits));
		rw.numWBit = 63 - numBits;
	}
}

void VolInputParser::writeRemainingBit2()
{
	if (rw.numWBit < 63) {
		//unsigned __int64 swapped = _byteswap_uint64(rw.wbyte);
		//writeData((unsigned char *)& swapped, sizeof(rw.wbyte));
		writeData((unsigned char *)& rw.wbyte, sizeof(rw.wbyte));
	}

}

uint64_t VolInputParser::readBit2(int to_read) {
    uint64_t result = 0;
    if (to_read <= rw.numRBit+1) {
        result = rw.rbyte << (63-rw.numRBit) >> (64-to_read);
		rw.numRBit -= to_read;
    }
    else {
        if (rw.numRBit > -1)
            result = rw.rbyte << (64- rw.numRBit-1) >> (64-to_read);
        readData(reinterpret_cast<uint8_t *> (&rw.rbyte), sizeof(rw.rbyte));
        to_read -= rw.numRBit +1;
        result |= rw.numRBit >> (64-to_read);
		rw.numRBit = 63-to_read;
    }
    return result;
}*/


//write the dimensions
void VolInputParser::writeCharacteristicData(int dim1, int dim2, int dim3, int U1R, int U1C, int U2R, int U2C, int U3R, int U3C)
{
	char txt[] = "erg.txt";
	BitIO::openWrite(txt);

	BitIO::writeBit(uint64_t(dim1), 32);
	BitIO::writeBit(uint64_t(dim2), 32);
	BitIO::writeBit(uint64_t(dim3), 32);

	BitIO::writeBit(uint64_t(U1R), 32);
	BitIO::writeBit(uint64_t(U1C), 32);
	
	BitIO::writeBit(uint64_t(U2R), 32);
	BitIO::writeBit(uint64_t(U2C), 32);
	
	BitIO::writeBit(uint64_t(U3R), 32);
	BitIO::writeBit(uint64_t(U3C), 32);

}

//read and safe dimensionalities of tensor for later calculations
void VolInputParser::readCharacteristicData()
{
	char txt[] = "erg.txt";
	BitIO::openRead(txt);

	tData.dim1 = BitIO::readBit(32);
	tData.dim2 = BitIO::readBit(32);
	tData.dim3 = BitIO::readBit(32);
	tData.U1R = BitIO::readBit(32);
	tData.U1C = BitIO::readBit(32);
	tData.U2R = BitIO::readBit(32);
	tData.U2C = BitIO::readBit(32);
	tData.U3R = BitIO::readBit(32);
	tData.U3C = BitIO::readBit(32);
}

//read and decode the encoded core and factor-matrices
void VolInputParser::readRleData(std::vector<std::vector<int>>& rle, std::vector<std::vector<bool>>& raw, double & scale, std::vector<bool>& signs)
{
	//first the scale
	int64_t scaleFaktor= BitIO::readBit(64);
	memcpy(&scale, (void*)&scaleFaktor, sizeof(scale));

	//second raw
	int rawSize = BitIO::readBit(64);
	for (int i = 0;i < rawSize;i++) {
		std::vector<bool> cur;
		int curSize = BitIO::readBit(64);
		for (int j = 0;j < curSize;j++) {
			bool c = BitIO::readBit(1);
			cur.push_back(c);
		}
		raw.push_back(cur);
	}

	//third rle
	TTHRESHEncoding::decodeACVektor(rle);
	//HuffmanCode coder;
	//coder.decodeData(rle);

	//fourth signs
	int signsSize = BitIO::readBit(64);
	for (int i = 0;i < signsSize;i++) {
		bool c = BitIO::readBit(1);
		signs.push_back(c);
	}

}

//read and safe norm-data for later decoding
void VolInputParser::readNormData()
{
	for (int i = 0; i < 3; i++) {
		uint64_t numElem = BitIO::readBit(64);
		std::vector<double> cur;
		for (int j = 0;j < numElem;j++) {
			double val;
			uint64_t readVal = BitIO::readBit(64);
			memcpy(&val, (void*)&readVal, sizeof(readVal));
			cur.push_back(val);
		}
		tData.coreSliceNorms.push_back(cur);
	}
}

