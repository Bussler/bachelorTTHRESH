#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <map>
#include "VolInputParser.h"
#include "TensorOperations.h"

namespace TTHRESHEncoding {

	//Thresholds for the AC algorithm
	extern int ACValueBits;
	extern uint64_t max;
	extern uint64_t oneFourth;
	extern uint64_t half;
	extern uint64_t threeFourth;

	enum ErrorType
	{
		epsilon,
		rmse,
		psnr
	};

	unsigned * getBits(uint64_t* n, int k, int numBits);
	
	std::vector<uint64_t> encodeRLE(double * coefficients, int numC, double errorTarget, std::vector<std::vector<int>> & rle, std::vector<std::vector<bool>>& raw, double& scale, std::vector<bool>& signs);
	double* decodeRLE(std::vector<std::vector<int>> rle, std::vector<std::vector<bool>> raw, int numC, double scale, std::vector<bool> signs);
	
	void compress(Eigen::Tensor<myTensorType, 3> b, std::vector<Eigen::MatrixXd> us, double errorTarget, ErrorType etype, std::vector<std::vector<int>> & rle, std::vector<std::vector<bool>>& raw, double& scale, std::vector<bool>& signs);

	//TODO maybe better to encode whole rle vector<vector> at once
	void encodeAC(std::vector<int> rle);
	void putBitPlusPending(bool bit, int& pending);

	void decodeAC(std::vector<int>& rle);

}


