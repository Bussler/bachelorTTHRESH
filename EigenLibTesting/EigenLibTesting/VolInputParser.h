#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <fstream>
#include "TensorOperations.h"


class VolInputParser
{
public:

	struct ReadWriteWrapper
	{
		uint64_t wbyte = 0; //store for 8 byte
		int numWBit = 63; //indicates free bit

		FILE * wFile;
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

	void writeCharacteristicData(int dim1, int dim2, int dim3, double scale);

};

