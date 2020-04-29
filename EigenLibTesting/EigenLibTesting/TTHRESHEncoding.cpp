#include "TTHRESHEncoding.h"
#include <fstream>
#define _USE_MATH_DEFINES

//stores the k-th bit of all n in bit-array
unsigned * TTHRESHEncoding::getBits(uint64_t* n, int k, int numBits)
{
	unsigned* bits = (unsigned*)malloc(sizeof(unsigned) * numBits);

	for (int i = 0; i < numBits; i++) {
		bits[i] = ((n[i] >> k) & 1); //right shifting desired bit and masking with 1
	}
	
	return bits;
}

//encode the coefficients with the help of rle/verbatim until error is below given threshold. Results are safed in rle and raw vectors
void TTHRESHEncoding::encodeRLE(double * c, int numC, double errorTarget, std::vector<std::vector<int>>& rle, std::vector<std::vector<bool>>& raw, double& scale, std::vector<bool>& signs)
{
	double max = 0;
	for (int i = 0;i < numC;i++) {
		if (abs(c[i])>max) {
			max = abs(c[i]);
		}
	}
	
	double scaleK = ldexp(1, 63 - ilogb(max));//calculate scale factor according to tthresh paper formula
	long double frobNormSq = 0;//Squared Frobeniusnorm of tensor
	
	uint64_t * n = (uint64_t*) malloc(sizeof(uint64_t)*numC);//array to store the scaled coefficients
	
	//cast coefficients from double to uint64 and calculate frobNormSq
	for (int i = 0; i < numC; i++) {
		n[i] = uint64_t(abs(c[i])*scaleK);//scale so that the leading bit of max element is 1
		frobNormSq += long double ( abs(c[i]) * abs(c[i]));
	}

	frobNormSq *= scaleK * scaleK;

	long double sse = frobNormSq;//exit condition: sse < given threshold
	long double thresh = errorTarget*errorTarget*frobNormSq;
	bool done = false;

	std::vector<uint64_t> mask(numC, 0);//creating bitmask to determine already relevant coefficients
	
	for (int p = 63;p >= 0; p--) {//running through bit-planes

		std::vector<bool> cRaw;
		std::vector<int> cRLE;
		int run = 0;
		int planeOnes = 0;//sse is reduced by each 1 in plane
		long double planeSSE = 0;

		for (int co = 0; co < numC; co++) {//running through the coefficients

			unsigned curBit = ((n[co] >> p) & 1);//access p-th bits
			planeOnes += curBit;

			if (mask[co]==0) {//not active, encode rle

				if (curBit==1) {
					cRLE.push_back(run);
					run = 0;
					//mask[co] = 1<<p;//mark in the mask where the leftmost leading bit was
				}
				else
				{
					run++;
				}

			}
			else {//is active, verbatim enociding
				cRaw.push_back(curBit>0);
			}

			if (curBit == 1) {//update sse and check for exit condition
				planeSSE += long double(n[co] - mask[co]);

				mask[co] |= (unsigned long long(1) << p);//mark in mask if active

				long double k = (unsigned long long(1) << p);
				long double sseCur = sse + (-2 * k*planeSSE + k * k*planeOnes);

				if (sseCur <= thresh) {
					done = true;
					std::cout << "End on Plane: " << p << " Coefficient: "<<co << std::endl;
					break;
				}

			}

		}

		cRLE.push_back(run); //safe last run

		rle.push_back(cRLE);
		raw.push_back(cRaw);

		long double k = (unsigned long long(1) << p);
		sse += (-2)*k*planeSSE + k * k*planeOnes;//updating sse if exit condition was not reached

		if (done) {
			break;
		}
	}

	scale = scaleK; //saving scale factor
	
	//saving signs
	for (int i = 0;i < numC;i++) {
		if (mask[i] > 0)
			signs.push_back(c[i] < 0); //if bit is set, the sign is negative
	}


	//TODO: Sizes der rle, raw vectoren speichern um wieder korrekt daten auslesen zu können

}

double * TTHRESHEncoding::decodeRLE(std::vector<std::vector<int>> rle, std::vector<std::vector<bool>> raw, int numC, double scale, std::vector<bool> signs)
{
	//convert back from rle/verbatim
	std::vector<uint64_t> mask(numC, 0);//holds the bits for coefficients
	int vecCount = 0;
	bool done = false;

	for (int p = 63; p >= 0; p--) {//for every bitplane

		int rleCount = 0;
		int verbCount = 0;

		int run = 0;
		if(rle[vecCount].size()>0)
		run = rle[vecCount][rleCount++];//current length of 0-run

		for (int co = 0;co < numC;co++) {//for every coefficient

			if (mask[co] == 0) {//rle encoding

				if (run == 0) {
					mask[co] |= (unsigned long long(1) << p); //encode 1
					
					if (rleCount < rle[vecCount].size())
						run = rle[vecCount][rleCount++]; //get next 0 run
				}
				else {
					run--;
				}

			}
			else {//verbatim encoding 
				if (raw[vecCount][verbCount]==true) {//==1
					mask[co] |= (unsigned long long(1) << p); //encode 1
				}
				verbCount++;
			}

			if (vecCount == rle.size()-1 && verbCount >= raw[vecCount].size() && rleCount >= rle[vecCount].size() ) {//stop the decription, if there is no data left
				done = true;
				break;
			}

		}

		vecCount++;

		if (done ||  vecCount==rle.size()) {
			break;
		}
	}

	//scale back into original double values, apply signs
	int signCount = 0;
	double * c = (double *) malloc(sizeof(double)*numC);
	for (int co = 0; co < numC; co++) {
		c[co] = double(mask[co] / scale);

		if (c[co] > 0 && signs[signCount++])//we apply a sign here, since coefficient is active
			c[co] *= (-1);
	}

	return c;
}

//convert sse according to specified errorType and starts the rle/verbatim encoding process
void TTHRESHEncoding::compress(double * coefficients, int numC, double errorTarget, ErrorType etype, std::vector<std::vector<int>>& rle, std::vector<std::vector<bool>>& raw, double& scale, std::vector<bool>& signs)
{
	//convert sse according to target error

	double dataNorm = 0;
	for (int i = 0; i < numC; i++) {
		dataNorm += (coefficients[i] * coefficients[i]);
	}
	dataNorm = sqrt(dataNorm);

	double sse;
	switch (etype)
	{
	case epsilon: sse = pow(errorTarget*dataNorm,2); //see paper for conversion formula
		break;

	case rmse: sse = pow(errorTarget,2)*numC;
		break;
	
	case psnr: //TODO
		break;
	}

	double convertedError = sqrt(sse) / dataNorm;

	encodeRLE(coefficients, numC, convertedError,rle, raw,scale, signs);

}

//encode the rle with AC Bit-Plane wise
std::vector<uint64_t> TTHRESHEncoding::encodeAC(std::vector<int> rle)
{
	//creating frequency/Interval Table
	std::map<uint64_t, std::pair<uint64_t, uint64_t>> freq;// key -> (count of key, lower bound Interval)
	for (int i = 0; i < rle.size(); i++) {
		freq[rle[i]].first += 1; //count the occurences of the key
	}

	uint64_t count = 0;
	for (auto it = freq.begin(); it != freq.end(); it++) {//calculate the probabilities and size of interval
		(it->second).second = count;
		count += ((it->second).first) / rle.size();//map to interval (0,1]
	}

	return std::vector<uint64_t>();
}
