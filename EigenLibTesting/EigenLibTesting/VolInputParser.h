#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <fstream>
#include "TensorOperations.h"
#include "TTHRESHEncoding.h"
#include <intrin.h>
#include "HuffmanCode.h"

namespace BitIO {

	extern struct RWWrapper rw; //wrapper class to hold information for read/write

	void writeData(unsigned char* data, int numBytes);
	void writeBit(uint64_t bits, int numBits);
	void writeRemainingBit();

	void readData(uint8_t * buf, int numBytes);
	uint64_t readBit(int numBits);

	void openRead(char* name);
	void closeRead();
	void openWrite(char* name);
	void closeWrite();

}

class VolInputParser
{
public:

	struct ReadTensorData {

		int dim1=0;
		int dim2=0;
		int dim3=0;
		int U1R=0;
		int U1C=0;
		int U2R=0;
		int U2C=0;
		int U3R=0;
		int U3C=0;

		std::vector<std::vector<double>> coreSliceNorms;

	}tData;

	VolInputParser();
	VolInputParser(char * txtname);
	~VolInputParser();

	//Eigen::MatrixXd MatrixData; //Matrix to store the data
	Eigen::Tensor<myTensorType, 3> TensorData; //Tensor to strore the data

	Eigen::Tensor<myTensorType, 3> DummyTensor;//dummy tensor for testing
	
	void readInputVol(char * txtname);

	void writeCharacteristicData(int dim1, int dim2, int dim3, int U1R, int U1C, int U2R, int U2C, int U3R, int U3C);
	void readCharacteristicData();
	void readRleData(std::vector<std::vector<int>>& rle, std::vector<std::vector<bool>>& raw, double& scale, std::vector<bool>& signs);
	void readNormData();

	/*void writeBit2(uint64_t bits, int numBits);
	void writeRemainingBit2();
	uint64_t readBit2(int to_read);*/
};

