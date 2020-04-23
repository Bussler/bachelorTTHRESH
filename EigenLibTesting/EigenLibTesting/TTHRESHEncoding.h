#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

namespace TTHRESHEncoding {

	unsigned * getBits(uint64_t* n, int k, int numBits);

	std::vector<int> RLE(unsigned * input, int numBits);
	
	void encodeRLE(double * coefficients, int numC, double errorTarget, std::vector<std::vector<int>> & rle, std::vector<std::vector<int>>& raw);
	
}


