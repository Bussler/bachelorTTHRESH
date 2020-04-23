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

//takes bit-array of size numBits as input, encodes it with rle
std::vector<int> TTHRESHEncoding::RLE(unsigned * input, int numBits)
{
	std::vector<int> rle;
	int run = 0;

	for (int i = 0;i < numBits; i++) {
		if (input[i]==1 || i==numBits-1) {
			rle.push_back(run);
			run = 0;
		}
		else {
			run++;
		}
	}

	return rle;
}


void TTHRESHEncoding::encodeRLE(double * c, int numC, double errorTarget, std::vector<std::vector<int>>& rle, std::vector<std::vector<int>>& raw)
{
	double max = 0;
	for (int i = 0;i < numC;i++) {
		if (abs(c[i])>max) {
			max = abs(c[i]);
		}
	}
	
	double scaleK = ldexp(1, 63 - ilogb(max));//calculate scale factor according to tthresh paper formula
	long double frobNormSq = 0;//Squared Frobeniusnorm of tensor, needed for sse TODO should be long long double
	
	uint64_t * n = (uint64_t*) malloc(sizeof(uint64_t)*numC);//array to store the scaled coefficients
	
	//cast coefficients from double to uint64 and calculate frobNormSq
	for (int i = 0; i < numC; i++) {
		n[i] = uint64_t(abs(c[i])*scaleK);//scale so that the leading bit of max element is 1
		frobNormSq += long double ( abs(c[i]) * abs(c[i]));
	}

	frobNormSq *= scaleK * scaleK;

	long double sse = frobNormSq;//exit condition: sse < given threshold TODO: shoul be long long double
	long double thresh = errorTarget*errorTarget*frobNormSq;
	bool done = false;

	std::cout << "SSE: " << sse << " Thresh: " << thresh << std::endl;

	std::vector<uint64_t> mask(numC, 0);//creating bitmask to determine already relevant coefficients
	
	for (int p = 63;p >= 0; p--) {//running through bit-planes

		std::vector<int> cRaw;
		std::vector<int> cRLE;
		int run = 0;
		int planeOnes = 0;//sse is reduced by each 1 in plane
		long double planeSSE = 0;

		for (int co = 0; co < numC; co++) {//running through the coefficients

			unsigned curBit = ((n[co] >> p) & 1);//access p-th bit
			planeOnes += curBit;

			if (mask[co]==0) {//not active, encode rle

				if (co==numC-1 && curBit==0) {//special case: end of line TODO: effizienter lösen, in dem ausßerhalb der for schleife gesetzt
					cRLE.push_back(run + 1);
				}

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
				cRaw.push_back(curBit);
			}

			if (curBit == 1) {//update sse and check for exit condition
				planeSSE += long double(n[co] - mask[co]);

				mask[co] |= 1 << p;//mark in mask if active

				long double k = (unsigned long long(1) << p); //TODO 1<<p ist zu groß für double
				long double sseCur = sse + (-2 * k*planeSSE + k * k*planeOnes);

				std::cout << "Plane sse: " << planeSSE << " Plane Ones: " << planeOnes << " sseCur: " << sseCur << " K: "<<k << std::endl;

				if (sseCur <= thresh) {
					done = true;
					std::cout << "End on Plane: " << p << " Coefficient: "<<co << std::endl;
					break;
				}

			}

		}

		rle.push_back(cRLE);
		raw.push_back(cRaw);

		long double k = (unsigned long long(1) << p);
		sse += (-2)*k*planeSSE + k * k*planeOnes;//updating sse if exit condition was not reached

		if (done) {
			break;
		}
	}


}
