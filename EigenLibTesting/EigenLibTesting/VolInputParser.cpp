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


/*//method to safe data bitwise //TODO von vorne nach hinten also 63...0 bit codieren nicht anders herum!
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

void VolInputParser::writeBit(uint64_t bits, int numBits)
{

	if (numBits+rw.vergWBit <= 64) {//we have free bit, just write them in TODO -1?
		rw.wbyte |= bits << (rw.vergWBit);
		rw.vergWBit += numBits;
	}
	else {// write as many bit as we can, store the rest again

		if (rw.vergWBit<64) {//if there is still something free, squeeze it in
			rw.wbyte |= (bits << (rw.vergWBit));
		}
		
		writeData((unsigned char *)& rw.wbyte, sizeof(rw.wbyte));

		numBits -= 64-rw.vergWBit;
		rw.wbyte = 0;
		rw.wbyte |= (bits >> (64 - rw.vergWBit));
		rw.vergWBit = 0+numBits;
	}
}

void VolInputParser::writeRemainingBit()
{
	if (rw.vergWBit >0 ) {
		writeData((unsigned char *)& rw.wbyte, sizeof(rw.wbyte));
	}

}

void VolInputParser::readData(uint8_t * buf, int numBytes)
{
	fread(buf, 1, numBytes, rw.rFile);
}

uint64_t VolInputParser::readBit(int numBits)
{
	uint64_t result = 0;
	if (numBits+rw.readRBit <=64) {//we haven't read everything from the buffer
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
		result |= rw.rbyte << amtShift >> (amtShift + 0) << (64-rw.readRBit);//shift to the amount of already read in data

		rw.readRBit = 0 + numBits;
	}

	return result;
}


//write the dimensions as short(), scale as double 
void VolInputParser::writeCharacteristicData(int dim1, int dim2, int dim3, double scale)
{
	rw.wFile = fopen("erg.txt", "w"); //Open document to write into

	writeBit(uint64_t(dim1), sizeof(unsigned short)*8);
	writeBit(uint64_t(dim2), sizeof(unsigned short) * 8);
	writeBit(uint64_t(dim3), sizeof(unsigned short) * 8);

	uint64_t tmp;
	memcpy(&tmp, (void*)&scale, sizeof(scale));
	writeBit(tmp, 64);

	writeRemainingBit();

	fclose(rw.wFile);

}

