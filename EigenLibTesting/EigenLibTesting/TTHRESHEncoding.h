#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <map>
#include "VolInputParser.h"

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
	
	void encodeRLE(double * coefficients, int numC, double errorTarget, std::vector<std::vector<int>> & rle, std::vector<std::vector<bool>>& raw, double& scale, std::vector<bool>& signs);
	double* decodeRLE(std::vector<std::vector<int>> rle, std::vector<std::vector<bool>> raw, int numC, double scale, std::vector<bool> signs);
	
	void compress(double * coefficients, int numC, double errorTarget, ErrorType etype, std::vector<std::vector<int>> & rle, std::vector<std::vector<bool>>& raw, double& scale, std::vector<bool>& signs);

	//TODO überarbeiten! decode...
	void encodeAC(std::vector<int> rle, VolInputParser& inParser);
	void putBitPlusPending(bool bit, int& pending, VolInputParser& inParser);

	void decodeAC(std::vector<int>& rle, VolInputParser& inParser);

}


