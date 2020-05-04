#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <fstream>
#include "TensorOperations.h"
#include <intrin.h>


class VolInputParser
{
public:

	struct ReadWriteWrapper
	{
		uint64_t wbyte = 0; //store for 8 byte
		uint64_t rbyte = 0;

		int numWBit = 63; //indicates free bit
		int numRBit = -1;

		int vergWBit = 0;
		int readRBit = 64;

		FILE * wFile;
		FILE * rFile;
	}rw;

	VolInputParser();
	VolInputParser(char * txtname);
	~VolInputParser();

	//Eigen::MatrixXd MatrixData; //Matrix to store the data
	Eigen::Tensor<myTensorType, 3> TensorData; //Tensor to strore the data

	Eigen::Tensor<myTensorType, 3> DummyTensor;//dummy tensor for testing
	
	void readInputVol(char * txtname);

	void writeData(unsigned char* data, int numBytes);
	void writeBit(uint64_t bits, int numBits);
	void writeRemainingBit();

	void readData(uint8_t * buf, int numBytes);
	uint64_t readBit(int numBits);

	void writeCharacteristicData(int dim1, int dim2, int dim3, double scale);
	void writeACData(std::vector<std::vector<int>> rle);

	/*void writeBit2(uint64_t bits, int numBits);
	void writeRemainingBit2();
	uint64_t readBit2(int to_read);*/
};

