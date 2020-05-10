#include "TTHRESHEncoding.h"
#include <fstream>
#define _USE_MATH_DEFINES

//Thresholds for the AC algorithm
int TTHRESHEncoding::ACValueBits = 32;
uint64_t TTHRESHEncoding::max = (unsigned long long(1) << TTHRESHEncoding::ACValueBits) - 1;//high value of range, can only decrease
uint64_t TTHRESHEncoding::oneFourth = (TTHRESHEncoding::max + 1) / 4;
uint64_t TTHRESHEncoding::half = TTHRESHEncoding::oneFourth*2;
uint64_t TTHRESHEncoding::threeFourth = TTHRESHEncoding::oneFourth * 3;

double price = -1, totalBitsCore= -1, errorCore=-1;//values for alpha calc
int encodingBitsAc = 0;

//stores the k-th bit of all n in bit-array
unsigned * TTHRESHEncoding::getBits(uint64_t* n, int k, int numBits)
{
	unsigned* bits = (unsigned*)malloc(sizeof(unsigned) * numBits);

	for (int i = 0; i < numBits; i++) {
		bits[i] = ((n[i] >> k) & 1); //right shifting desired bit and masking with 1
	}
	
	return bits;
}

//encode the coefficients with the help of rle/verbatim until error is below given threshold. Results are safed in rle and raw vectors: Adapted from rballester Github (Alpha, SSE calc)
std::vector<uint64_t> TTHRESHEncoding::encodeRLE(double * c, int numC, double errorTarget, bool isCore, std::vector<std::vector<int>>& rle, std::vector<std::vector<bool>>& raw, double& scale, std::vector<bool>& signs)
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

	//values for alpha calc
	long double lastError = 1;
	int totalBits=0;
	int lastTotalBits = 0;
	double errorDelta = 0, sizeDelta = 0, error = 0;

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
				totalBits++;
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

		
		//TODO encode rle/raw here, don't safe them first into vectors
		totalBits += 64;//saving rawsize
		totalBits += 180;//saving rle, annahme: pro rle werden 254 bit gespeichert
		error = sqrt(double(sse/frobNormSq));
		if (lastTotalBits>0) {
			if (isCore) {
				sizeDelta = (totalBits-lastTotalBits)/double(lastTotalBits);
				errorDelta = (lastError - error) / error;
			}
			else {//exit loop for factor matrizes
				if ((totalBits/totalBitsCore)/(error/errorCore) >= price) { //reduction in sse by last co bits / number if bits to compress <= core-alpha
					std::cout << "yeetus deletus" << std::endl;
					done = true;
				}
			}
		}
		lastTotalBits = totalBits;
		lastError = error;

		if (done) {
			break;
		}
	}

	scale = scaleK; //saving scale factor in var
	uint64_t tmp;
	memcpy(&tmp, (void*)&scale, sizeof(scale));
	BitIO::writeBit(tmp, 64);

	//save raw
	BitIO::writeBit(raw.size(), 64);//overall size
	for (int i = 0; i < raw.size();i++) {
		BitIO::writeBit(raw[i].size(), 64);//size of array
		for (int j = 0;j < raw[i].size();j++) {
			BitIO::writeBit(raw[i][j], 1);//saving data
		}
	}

	//save rle
	BitIO::writeBit(rle.size(), 64);//overall size
	for (int i = 0; i < rle.size();i++) {
		encodeAC(rle[i]);//encode rle with ac
	}

	//saving signs
	for (int i = 0;i < numC;i++) {
		if (mask[i] > 0)
			signs.push_back(c[i] < 0); //if bit is set, the sign is negative
	}
	BitIO::writeBit(signs.size(),64);//saving size
	for (int i = 0;i < signs.size();i++) {
		BitIO::writeBit(signs[i], 1);//saving data
		totalBits++;//TODO also safe size, so +64 at beginning?
	}

	//TODO: Encoded Bit innerhalb berechnen: wie viele bit werden innerhalb der schleife für rle etc benötigt? Und wie kann rle, raw innerhalb der schleife geschrieben werden und gleichzeitig size der vektoren gespeichert werden?

	if (isCore) {//safe alpha values for core
		price = sizeDelta / errorDelta;
		errorCore = error;
		totalBitsCore = totalBits;
	}

	return mask;
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
void TTHRESHEncoding::compress(Eigen::Tensor<myTensorType, 3> b, std::vector<Eigen::MatrixXd> us, double errorTarget, ErrorType etype, std::vector<std::vector<int>>& rle, std::vector<std::vector<bool>>& raw, double& scale, std::vector<bool>& signs)
{
	double* coefficients = b.data();
	int numC = b.dimension(0)*b.dimension(1)*b.dimension(2);
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

	std::vector<uint64_t> CoreMask = encodeRLE(coefficients, numC, convertedError, true, rle, raw,scale, signs); // encode the core

	//encode the factor matrices: calculate core-slice norms TODO rballester special case 0
	Eigen::Tensor<myTensorType, 3> maskTensor = b;// TensorOperations::createTensorFromArray((myTensorType*)CoreMask.data(), b.dimension(0), b.dimension(1), b.dimension(2));

	std::vector<std::vector<double>> usNorms;

	for (int i = 0; i < us.size();i++) {//multiply each U col with core-slice norm
		std::vector<double> n;
		for (int j = 0;j < us[i].cols();j++) {
			
			Eigen::MatrixXd slice = TensorOperations::getSlice(maskTensor, i+1, j);
			us[i].col(j) *= slice.norm();
			n.push_back(slice.norm());
			//std::cout << j << ": " << slice.norm()<<std::endl<<slice<<std::endl;
		}
		usNorms.push_back(n);
	}
	//std::cout << "U1 nach Scale: " << std::endl << us[0] << std::endl;
	std::cout << "Num Norms: " << usNorms.size()<<std::endl;
	BitIO::writeBit(usNorms.size(), 64);
	/*for (int i = 0;i < usNorms.size();i++) {//TODO
		uint64_t tmp;
		memcpy(&tmp, (void*)&usNorms[i], sizeof(usNorms[i]));
		BitIO::writeBit(tmp, 64);//slicenorms abspeichern
	}*/


	std::vector<std::vector<std::vector<int>>> usRle;
	std::vector<std::vector<std::vector<bool>>> usRaw;
	std::vector<double> usScales;
	std::vector<std::vector<bool>> usSigns;
	
	for (int i = 0; i < us.size(); i++) {//encode the factor matizes
		usRle.push_back(std::vector < std::vector<int>>());
		usRaw.push_back(std::vector < std::vector<bool>>());
		usScales.push_back(0);
		usSigns.push_back(std::vector<bool>());

		encodeRLE(us[i].data(), us[i].cols()*us[i].rows(), convertedError,false, usRle[i], usRaw[i], usScales[i], usSigns[i] );
	}

	//Test for decoding
	int kek = 0;
	double * decU = decodeRLE(usRle[kek], usRaw[kek], us[kek].cols()*us[kek].rows(), usScales[kek], usSigns[kek]);
	Eigen::MatrixXd decMU = TensorOperations::createMatrixFromArray(decU, us[kek].rows(), us[kek].cols());
	for (int j = 0;j < us[kek].cols();j++) {
		decMU.col(j) /= usNorms[kek][j];
	}
	std::cout << "Decoded Factor"<<kek<< ": " << std::endl << decMU << std::endl;

}

//encode the rle with AC Bit-Plane wise: Taken from rballester Github
void TTHRESHEncoding::encodeAC(std::vector<int> rle)
{
	encodingBitsAc = 0;

	//creating frequency/Interval Table : The model
	std::map<uint64_t, std::pair<uint64_t, uint64_t>> freq;// key -> (count of key, lower bound Interval)
	for (int i = 0; i < rle.size(); i++) {
		freq[rle[i]].first += 1; //count the occurences of the key
	}

	uint64_t count = 0;
	for (auto it = freq.begin(); it != freq.end(); it++) {//calculate the probabilities and size of interval
		(it->second).second = count;
		count += ((it->second).first);// integer arithmetic, so don't map to [0,1)
	}

	//saving the model for later decode
	uint64_t freqSize = freq.size();
	BitIO::writeBit(freqSize, 64);
	encodingBitsAc += 64;

	for (auto it = freq.begin(); it != freq.end(); it++) {
		uint64_t key = it->first;
		uint64_t prob = (it->second).first;

		//encode key len, then key to safe space
		uint8_t keyLen = 0;
		uint64_t keyCopy = key;
		while (keyCopy>0) {
			keyCopy >>= 1;
			keyLen++;
		}
		if (keyLen == 0)
			keyLen = 1;
		
		BitIO::writeBit(keyLen, 6);
		BitIO::writeBit(key, keyLen);//save key 

		uint8_t probLen = 0;
		uint64_t probCopy = prob;
		while (probCopy > 0) {
			probCopy >>= 1;
			probLen++;
		}
		if (probLen == 0)
			probLen = 1;
		
		BitIO::writeBit(probLen, 6);
		BitIO::writeBit(prob, probLen);//save prob

		encodingBitsAc += 6 + 6 + keyLen + probLen;

		//BitIO::writeBit(key, 32);//save key and frequenzy with 32 bit
		//BitIO::writeBit(prob, 32);
	}

	uint64_t rleSize = rle.size();
	BitIO::writeBit(rleSize, 64);//save the number of symbols to encode
	encodingBitsAc += 64;

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
				putBitPlusPending(0,pendingBits);
			}
			else if (low >= half) {//MSB==1
				putBitPlusPending(1, pendingBits);
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
		putBitPlusPending(0, pendingBits);
	}
	else {
		putBitPlusPending(1, pendingBits);
	}

	//Write trailing 0s
	BitIO::writeBit(0, ACValueBits- 2);
	encodingBitsAc += ACValueBits - 2;//TODO encoding bits in rle einbinden!
	//std::cout << "AC Bit: " << encodingBitsAc << std::endl;
}

void TTHRESHEncoding::putBitPlusPending(bool bit, int & pending)
{
	BitIO::writeBit(bit,1);
	encodingBitsAc++;
	for (int i = 0;i < pending;i++) {
		BitIO::writeBit(!bit,1);
		encodingBitsAc++;
	}
	pending = 0;
}

//decode algorithm of ac: taken from rballester Github
void TTHRESHEncoding::decodeAC(std::vector<int>& rle)
{
	//read and recreate the saved frequenzy table
	uint64_t freqSize = BitIO::readBit(64); //table size safed with 64 bit

	std::map<uint64_t, uint64_t> freq;//lower frequenzy -> key
	uint64_t count = 0;

	for (int i = 0;i < freqSize;i++) {
		//uint64_t key = BitIO::readBit(32);
		//uint64_t prob = BitIO::readBit(32);
		uint64_t keyLen = BitIO::readBit(6);
		uint64_t key = BitIO::readBit(keyLen);
		uint64_t probLen = BitIO::readBit(6);
		uint64_t prob = BitIO::readBit(probLen);
		
		freq[count] = key;
		count += prob;
	}

	uint64_t rleSize = BitIO::readBit(64);
	if (rleSize == 0) {//TODO special case if we encoded a rle with size 0
		for (int i = 0;i < ACValueBits;i++) {
			int h= BitIO::readBit(1);
		}
		return;
	}

	freq[rleSize] = 0;//we need another upper bound

	//decoding
	uint64_t high = max;
	uint64_t low = 0;
	uint64_t val = 0;//= inParser.readBit(ACValueBits); //encoded value to decode
	for (int i = 0;i < ACValueBits;i++) { //TODO problem with little endian read in, instead shift all bit one by one
		val <<= 1;
		val += BitIO::readBit(1) ? 1 : 0;
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
			val += BitIO::readBit(1) ? 1 : 0;

		}

		if (rle.size()== rleSize) {//we have all our decoded symbols
			break;
		}

	}

}



//try to encode and write the whole vektor with a single freq model
void TTHRESHEncoding::encodeACVektor(std::vector<std::vector<int>> rleVek)
{
	int wholeSize = 0;
	//creating frequency/Interval Table : The model
	std::map<uint64_t, std::pair<uint64_t, uint64_t>> freq;// key -> (count of key, lower bound Interval)
	for (int i = 0; i < rleVek.size(); i++) {
		for (int j = 0; j < rleVek[i].size(); j++) {
			freq[rleVek[i][j]].first += 1; //count the occurences of the key
			wholeSize++;
		}

	}

	uint64_t count = 0;
	for (auto it = freq.begin(); it != freq.end(); it++) {//calculate the probabilities and size of interval
		(it->second).second = count;
		count += ((it->second).first);// integer arithmetic, so don't map to [0,1)
	}

	//saving the model for later decode
	uint64_t freqSize = freq.size();
	BitIO::writeBit(freqSize, 64);

	for (auto it = freq.begin(); it != freq.end(); it++) {
		uint64_t key = it->first;
		uint64_t prob = (it->second).first;

		BitIO::writeBit(key, 32);//save key and frequenzy with 32 bit
		BitIO::writeBit(prob, 32);
	}

	uint64_t rleVekSize = rleVek.size();
	BitIO::writeBit(rleVekSize, 64); //save the number of elements to write

	for (int i = 0;i < rleVekSize;i++) {
		uint64_t rleSize = rleVek[i].size();
		BitIO::writeBit(rleSize, 64);//save the number of symbols to encode
	}
	
	//after model is built, encode input
	int pendingBits = 0; //Counter to store pending bits when low and high are converging
	uint64_t low = 0; //low limit for interval, can only increase
	uint64_t high = max; //high limit for interval, can only decrease

	for (int itOverRle = 0; itOverRle < rleVek.size();itOverRle++) { // iterate over bitplanes and encode them

		for (int i = 0;i < rleVek[itOverRle].size();i++) {
			uint64_t cur = rleVek[itOverRle][i];

			uint64_t probHigh = freq[cur].second + freq[cur].first;//gets top interval Value
			uint64_t probLow = freq[cur].second;//gets bottom interval value

			uint64_t newRange = high - low + 1; //current subset
			high = low + (newRange*probHigh / wholeSize) - 1; //progressive subdividing of range TODO groesse des ganzen arrays benutzen
			low = low + (newRange*probLow / wholeSize);

			while (true) {

				if (high < half) {//MSB==0
					putBitPlusPending(0, pendingBits);
				}
				else if (low >= half) {//MSB==1
					putBitPlusPending(1, pendingBits);
				}
				else if (low >= oneFourth && high < threeFourth) {//converging
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


	}

	pendingBits++;
	if (low < oneFourth) {
		putBitPlusPending(0, pendingBits);
	}
	else {
		putBitPlusPending(1, pendingBits);
	}

	//Write trailing 0s
	BitIO::writeBit(0, ACValueBits - 2);

}

void TTHRESHEncoding::decodeACVektor(std::vector<std::vector<int>>& rleVek)
{
	//read and recreate the saved frequenzy table
	uint64_t freqSize = BitIO::readBit(64); //table size safed with 64 bit

	std::map<uint64_t, uint64_t> freq;//lower frequenzy -> key
	uint64_t count = 0;

	for (int i = 0;i < freqSize;i++) {
		uint64_t key = BitIO::readBit(32);
		uint64_t prob = BitIO::readBit(32);

		freq[count] = key;
		count += prob;
	}

	uint64_t rleVekSize = BitIO::readBit(64);
	std::vector<int> rleSizes;
	int wholeSize = 0;
	for (int i = 0;i < rleVekSize;i++) {
		int curVekSize = BitIO::readBit(64);
		rleSizes.push_back(curVekSize);
		wholeSize += curVekSize;
	}

	freq[wholeSize] = 0;//we need another upper bound

		std::vector<int> curPlane;

		int rleVekCounter = 0;
		uint64_t rleSize = rleSizes[rleVekCounter++];

		//decoding
		uint64_t high = max;
		uint64_t low = 0;
		uint64_t val = 0; //encoded value to decode
		for (int i = 0;i < ACValueBits;i++) { //TODO problem with little endian read in, instead shift all bit one by one
			val <<= 1;
			val += BitIO::readBit(1) ? 1 : 0;
		}

		while (true) {

			if (rleSize != 0) {

				uint64_t range = high - low + 1;
				uint64_t scaledVal = ((val - low + 1)*wholeSize - 1) / range;

				auto it = freq.upper_bound(scaledVal);
				uint64_t pHigh = it->first;//high bound of interval
				it--;
				uint64_t pLow = it->first;//low bound of interval

				curPlane.push_back(int(it->second));//save the key, the decoded signal

				high = low + (range*pHigh) / wholeSize - 1;
				low = low + (range*pLow) / wholeSize;

				while (true) {

					if (high < half) {//bit is 0

					}
					else if (low >= half) {
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
					val += BitIO::readBit(1) ? 1 : 0;

				}

			}

			if (curPlane.size() == rleSize) {//we have all our decoded symbols
				rleVek.push_back(curPlane);

				if (rleVek.size() == rleVekSize) {
					break;
				}

				rleSize = rleSizes[rleVekCounter++];
				curPlane.clear();
			}

		}

}

