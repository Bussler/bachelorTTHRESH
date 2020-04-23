#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

namespace TTHRESHEncoding {

	enum ErrorType
	{
		epsilon,
		rmse
	};

	unsigned * getBits(uint64_t* n, int k, int numBits);

	std::vector<int> RLE(unsigned * input, int numBits);
	
	void encodeRLE(double * coefficients, int numC, double errorTarget, std::vector<std::vector<int>> & rle, std::vector<std::vector<int>>& raw);
	void decodeRLE();
	
	void compress(double * coefficients, int numC, double errorTarget, ErrorType etype, std::vector<std::vector<int>> & rle, std::vector<std::vector<int>>& raw);

}


