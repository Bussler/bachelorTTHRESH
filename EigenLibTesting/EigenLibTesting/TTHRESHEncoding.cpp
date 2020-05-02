#include "TTHRESHEncoding.h"
#include <fstream>
#define _USE_MATH_DEFINES

//Thresholds for the AC algorithm
int TTHRESHEncoding::ACValueBits = 32;
uint64_t TTHRESHEncoding::max = (unsigned long long(1) << TTHRESHEncoding::ACValueBits) - 1;//high value of range, can only decrease
uint64_t TTHRESHEncoding::oneFourth = (TTHRESHEncoding::max + 1) / 4;
uint64_t TTHRESHEncoding::half = TTHRESHEncoding::oneFourth*2;
uint64_t TTHRESHEncoding::threeFourth = TTHRESHEncoding::oneFourth * 3;

//stores the k-th bit of all n in bit-array
unsigned * TTHRESHEncoding::getBits(uint64_t* n, int k, int numBits)
{
	unsigned* bits = (unsigned*)malloc(sizeof(unsigned) * numBits);

	for (int i = 0; i < numBits; i++) {
		bits[i] = ((n[i] >> k) & 1); //right shifting desired bit and masking with 1
	}
	
	return bits;
}

//encode the coefficients with the help of rle/verbatim until error is below given threshold. Results are safed in rle and raw vectors: Taken from rballester Github
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

//encode the rle with AC Bit-Plane wise: Taken from rballester Github
void TTHRESHEncoding::encodeAC(std::vector<int> rle, VolInputParser& inParser)
{
	//creating frequency/Interval Table : The model
	std::map<uint64_t, std::pair<uint64_t, uint64_t>> freq;// key -> (count of key, lower bound Interval)
	for (int i = 0; i < rle.size(); i++) {
		freq[rle[i]].first += 1; //count the occurences of the key
	}

	uint64_t count = 0;
	for (auto it = freq.begin(); it != freq.end(); it++) {//calculate the probabilities and size of interval
		(it->second).second = count;
		count += ((it->second).first);// integer arithmetic, so don't map to [0,1)
		//std::cout << "Key: " << it->first << " Prob: " << (it->second).second<<std::endl;
	}

	//saving the model for later decode
	uint64_t freqSize = freq.size();
	inParser.writeBit(freqSize, 64);

	for (auto it = freq.begin(); it != freq.end(); it++) {
		uint64_t key = it->first;
		uint64_t prob = (it->second).first;

		inParser.writeBit(key, 32);//save key and frequenzy with 32 bit
		inParser.writeBit(prob, 32);
	}

	uint64_t rleSize = rle.size();
	inParser.writeBit(rleSize, 64);//save the number of symbols to encode

	//after model is built, encode input
	int pendingBits = 0; //Counter to store pending bits when low and high are converging
	uint64_t low = 0; //low limit for interval, can only increase
	uint64_t high = max; //high limit for interval, can only decrease

	for (int i = 0;i < rle.size();i++) {
		uint64_t cur = rle[i];

		uint64_t probHigh = freq[cur].second + freq[cur].first;//gets top interval Value
		uint64_t probLow = freq[cur].second;//gets bottom interval value

		uint64_t newRange = high - low + 1; //current subset
		high = low + (newRange*probHigh / rle.size()) -1; //progressive subdividing of range
		low = low + (newRange*probLow / rle.size());

		while (true) {

			if (high<half) {//MSB==0
				putBitPlusPending(0,pendingBits,inParser);
			}
			else if (low >= half) {//MSB==1
				putBitPlusPending(1, pendingBits, inParser);
			}
			else if (low>=oneFourth && high<threeFourth) {//converging
				pendingBits++;
				low -= oneFourth;
				high -= oneFourth;
			}
			else
			{
				break;
			}

			high <<= 1;
			high++;//high: never ending stream of ones, low: stream of zeroes
			low <<= 1;
			high &= max;
			low &= max;

		}

	}

	pendingBits++;
	if (low < oneFourth) {
		putBitPlusPending(0, pendingBits, inParser);
	}
	else {
		putBitPlusPending(1, pendingBits, inParser);
	}

	//Write trailing 0s
	inParser.writeBit(0, ACValueBits - 2);
}

void TTHRESHEncoding::putBitPlusPending(bool bit, int & pending, VolInputParser& inParser)
{
	inParser.writeBit(bit,1);
	for (int i = 0;i < pending;i++) {
		inParser.writeBit(!bit,1);
	}
	pending = 0;
}

//decode algorithm of ac: taken from rballester Github
void TTHRESHEncoding::decodeAC(std::vector<int>& rle, VolInputParser & inParser)
{
	//read and recreate the saved frequenzy table
	uint64_t freqSize = inParser.readBit(64); //table size safed with 64 bit

	std::map<uint64_t, uint64_t> freq;//lower frequenzy -> key
	uint64_t count = 0;

	for (int i = 0;i < freqSize;i++) {
		uint64_t key = inParser.readBit(32);
		uint64_t prob = inParser.readBit(32);
		
		freq[count] = key;
		std::cout << "Key: " << key << " Prob: " << count<<std::endl;
		count += prob;
	}

	uint64_t rleSize = inParser.readBit(64);
	freq[rleSize] = 0;//we need another upper bound

	//decoding
	uint64_t high = max;
	uint64_t low = 0;
	uint64_t val = 0;//= inParser.readBit(ACValueBits); //encoded value to decode
	for (int i = 0;i < ACValueBits;i++) { //TODO problem with little endian read in, instead shift all bit one by one
		val <<= 1;
		val += inParser.readBit(1) ? 1 : 0;
	}

	while (true) {

		uint64_t range = high - low + 1;
		uint64_t scaledVal = ((val-low+1)*rleSize-1)/range;

		auto it = freq.upper_bound(scaledVal);
		uint64_t pHigh = it->first;//high bound of interval
		it--;
		uint64_t pLow = it->first;//low bound of interval

		rle.push_back(int(it->second));//save the key, the decoded signal

		high = low + (range*pHigh) / rleSize - 1;
		low = low + (range*pLow) / rleSize;

		while (true) {

			if (high<half) {//bit is 0

			}
			else if (low>=half) {
				val -= half;
				low -= half;
				high -= half;
			}
			else if (low >= oneFourth && high < threeFourth) {
				val -= oneFourth;
				low -= oneFourth;
				high -= oneFourth;
			}
			else {
				break;
			}

			low <<= 1;
			high <<= 1;
			high++;//high: never ending stream of ones
			val <<= 1;
			val += inParser.readBit(1) ? 1 : 0;

		}

		if (rle.size()== rleSize) {//we have all our decoded symbols
			break;
		}

	}

}
