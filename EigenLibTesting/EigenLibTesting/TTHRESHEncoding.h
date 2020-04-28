#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>


namespace TTHRESHEncoding {

	enum ErrorType
	{
		epsilon,
		rmse,
		psnr
	};

	unsigned * getBits(uint64_t* n, int k, int numBits);
	
	void encodeRLE(double * coefficients, int numC, double errorTarget, std::vector<std::vector<int>> & rle, std::vector<std::vector<bool>>& raw, double& scale, std::vector<bool>& signs);
	double* decodeRLE(std::vector<std::vector<int>> rle, std::vector<std::vector<bool>> raw, int numC, double scale, std::vector<bool> signs);
	
	void compress(double * coefficients, int numC, double errorTarget, ErrorType etype, std::vector<std::vector<int>> & rle, std::vector<std::vector<bool>>& raw, double& scale, std::vector<bool>& signs);

}


