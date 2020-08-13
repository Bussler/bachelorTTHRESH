#include "TensorTruncation.h"


//quantizatization method to map val to 8 bit value between [0,255]
int TensorTruncation::logQuantize(double val, double max)
{
	return (255*log2(1+abs(val)))/(log2(1+abs(max)));
}

double TensorTruncation::logDequantize(int val, double max)//log2(16) = 4 -> 2^4 = 16 
{
	return pow(2,((val*(log2(1+abs(max))))/255))-1;
}

//performs quantization and maps numbers to 9 bits: 1 bit sign, 8 bit value
void TensorTruncation::QuantizeData(double * coefficients, int numC, bool isCore)
{
	//first find abs value of max element
	double max = 0;
	for (int i = 0;i < numC;i++) {
		if (abs(coefficients[i]) > max) {
			max = abs(coefficients[i]);
		}
	}

	uint64_t tmpMax;
	memcpy(&tmpMax, (void*)&max, sizeof(max));
	BitIO::writeBit(tmpMax, 64);//safe max element

	if (isCore) {//safe hot corner element seperately

		double hotCornerElement = coefficients[0];//safe hot corner seperately, since it holds most of the core's energy
		uint64_t tmp;
		memcpy(&tmp, (void*)&hotCornerElement, sizeof(hotCornerElement));
		BitIO::writeBit(tmp, 64);//safe hot corner element

		for (int i = 1;i < numC; i++) {
			int quant = logQuantize(coefficients[i], max);
			//write Data in 9 bit here
			BitIO::writeBit(coefficients[i] < 0, 1);//saving sign with 1 bit: 1 for negative
			BitIO::writeBit(quant, 8);//8 bit for quantization

			//std::cout << "Orig: " << coefficients[i] << " Quant: " << quant << " Reconstructed: "<<logDequantize(quant,max) << std::endl;
		}

	}
	else {

		for (int i = 0;i < numC; i++) {
			int quant = logQuantize(coefficients[i], max);
			//write Data in 9 bit here
			BitIO::writeBit(coefficients[i] < 0, 1);//saving sign with 1 bit: 1 for negative
			BitIO::writeBit(quant, 8);//8 bit for quantization
		}

	}

}

TensorTruncation::OptimalChoice TensorTruncation::createChoiceNode(double rE, double F, int r1, int r2, int r3)
{
	OptimalChoice oc;
	oc.rE = rE;
	oc.F = F;
	oc.r1 = r1;
	oc.r2 = r2;
	oc.r3 = r3;

	return oc;
}

bool compareByF(const TensorTruncation::OptimalChoice &a, const TensorTruncation::OptimalChoice &b)
{
	return a.F < b.F;
}

//iterate over core b and build Summead Area Table for fast computation of frobenius norm: save squared values for frob.Norm computation
void TensorTruncation::buildSummedAreaTable(Eigen::Tensor<myTensorType, 3>& b, Eigen::Tensor<myTensorType, 3>& SAT)
{
	for (int z = 0;z < b.dimension(2);z++) {
		for (int x = 0; x < b.dimension(1); x++) {
			for (int y = 0;y < b.dimension(0); y++) {

				//Grenzen überprüfen + Formel
				double ival = pow(b(y, x, z), 2);
				
				if (x > 0 && y > 0 && z > 0)
					ival += SAT(y - 1, x - 1, z - 1);
				if (z > 0)
					ival += SAT(y, x, z - 1);
				if (y > 0)
					ival += SAT(y - 1, x, z);
				if (x > 0)
					ival += SAT(y, x - 1, z);
				if (x > 0 && y > 0)
					ival -= SAT(y - 1, x - 1, z);
				if (y > 0 && z > 0)
					ival -= SAT(y - 1, x, z - 1);
				if (x > 0 && z > 0)
					ival -= SAT(y, x - 1, z - 1);

				SAT(y, x, z) = ival;

			}
		}
	}

}


void TensorTruncation::CalculateTruncation(Eigen::Tensor<myTensorType, 3>& b, std::vector<Eigen::MatrixXd>& us, double givenRe)
{
	//calculate r1,r2,r3 values

	//build summed area table for fast computation of frob norm: needed for relative error estimate
	Eigen::Tensor<myTensorType, 3> SAT(b.dimension(0), b.dimension(1), b.dimension(2));
	SAT.setZero();
	TensorTruncation::buildSummedAreaTable(b, SAT);

	//calculating all possibilities
	double frobNormSquared = SAT(b.dimension(0) - 1, b.dimension(1) - 1, b.dimension(2) - 1);
	double frobNorm = sqrt(frobNormSquared);
	double D123 = b.dimension(0)*b.dimension(1)*b.dimension(2);

	std::vector<OptimalChoice> choices;//data container to hold all choices for r1,r2,r3 and rel error, compression factor

	for (int r3 = 1; r3 <= b.dimension(2); r3++) {
		for (int r2 = 1; r2 <= b.dimension(1); r2++) {
			for (int r1 = 1; r1 <= b.dimension(0); r1++) {
				
				//formulas according to R.Ballester Paper "Lossy Volume Compression using Tucker truncation and thresholding"
				double F = ((r1*r2*r3) + (r1*b.dimension(0)) + (r2*b.dimension(1)) + (r3*b.dimension(2))) / D123;
				double relErr = sqrt(frobNormSquared - SAT(r1-1, r2-1, r3-1)) / frobNorm;

				choices.push_back(createChoiceNode(relErr, F, r1, r2, r3));

			}
		}
	}

	std::vector<OptimalChoice> C;//data container to hold all best choices for r1,r2,r3 and rel error, compression factor
	//sort choice-vector in increasing order (repsect to F)
	std::sort(choices.begin(), choices.end(), compareByF);
	double bestError = DBL_MAX;
	for (int i = 0;i < choices.size();i++) {
		
		if (choices[i].rE<bestError) {//found improvement
			bestError = choices[i].rE;
			C.push_back(choices[i]);
		}

	}

	//truncate core given the optimal r1, r2, r3 values
	int r1 = 0;//test for (2, 2, 2), (200, 200, 150)
	int r2 = 0;
	int r3 = 0;

	for (int i = 0;i < C.size();i++) {//find best re-f pair
		if (C[i].rE<=givenRe || i==C.size()-1) {
			if ((i>0) && abs(C[i].rE-givenRe)>abs(C[i-1].rE-givenRe)) {
				r1 = C[i-1].r1;
				r2 = C[i - 1].r2;
				r3 = C[i - 1].r3;
			}
			else {
				r1 = C[i].r1;
				r2 = C[i].r2;
				r3 = C[i].r3;
			}
			break;
		}
	}

	std::cout << "R: " << r1 << " " << r2 << " " << r3 << std::endl;

	BitIO::writeBit(uint64_t(r1), 32);
	BitIO::writeBit(uint64_t(r2), 32);
	BitIO::writeBit(uint64_t(r3), 32);

	truncateTensor(b, us, r1, r2, r3);

}

void TensorTruncation::CalculateRetruncation(Eigen::Tensor<myTensorType, 3>& b, std::vector<Eigen::MatrixXd>& us, int d1, int d2, int d3, int r1, int r2, int r3)
{
	//read in the truncated core
	myTensorType* truncatedCore = (myTensorType*)malloc(sizeof(myTensorType)*(r1*r2*r3)); //pointer to hold surviving core data

	int64_t hotTempMax = BitIO::readBit(64);//first read in the max element
	double max = 0;
	memcpy(&max, (void*)&hotTempMax, sizeof(max));


	int64_t hotTemp = BitIO::readBit(64);//second read in the hot corner element
	double hotCornerElement = 0;
	memcpy(&hotCornerElement, (void*)&hotTemp, sizeof(hotCornerElement));
	truncatedCore[0] = hotCornerElement;

	for (int i = 1;i < r1*r2*r3; i++) {//read in and dequantize the values
		bool sign = BitIO::readBit(1);
		int quant = BitIO::readBit(8);

		double dequantize = logDequantize(quant, max);

		if (sign) {
			truncatedCore[i] = (-1)* dequantize;
		}
		else {
			truncatedCore[i] = dequantize;
		}
		
	}

	b = TensorOperations::createTensorFromArray(truncatedCore, r1, r2, r3);

	//read in the truncated factor matrices
	for (int i = 0;i < 3;i++) {

		int cols = 0;
		int rows = 0;
		
		switch (i)
		{
		case 0:
			rows = d1;
			cols = r1;
			break;

		case 1:
			rows = d2;
			cols = r2;
			break;

		case 2:
			rows = d3;
			cols = r3;
			break;
		}

		double* data = (double*)malloc(sizeof(double)*(cols*rows)); //pointer to hold surviving matrix data

		int64_t hotTempMax = BitIO::readBit(64);//first read in the max element
		double max = 0;
		memcpy(&max, (void*)&hotTempMax, sizeof(max));

		for (int i = 0;i < cols*rows; i++) {
			bool sign = BitIO::readBit(1);
			int quant = BitIO::readBit(8);

			double dequantize = logDequantize(quant, max);

			if (sign) {
				data[i] = (-1)* dequantize;
			}
			else {
				data[i] = dequantize;
			}
		}

		us[i] = Eigen::Map<Eigen::MatrixXd>(data, rows, cols);


	}

	ReTruncateTensor(b, us, d1, d2, d3);//fill missing values with 0

}


//slice core and factor matrix till we have a (r1,r2,r3) core; rn <= dim of core
void TensorTruncation::truncateTensor(Eigen::Tensor<myTensorType, 3>& b, std::vector<Eigen::MatrixXd>& us, int r1, int r2, int r3)
{
	myTensorType* truncatedCore= (myTensorType*)malloc(sizeof(myTensorType)*(r1*r2*r3)); //pointer to hold surviving core data

	//truncating the core
	int coreCounter = 0;
	for (int z = 0;z < r3; z++) {
		for (int x = 0;x < r2; x++) {
			for (int y = 0;y < r1; y++) {
				truncatedCore[coreCounter++] = b(y, x, z);
			}
		}
	}

	QuantizeData(truncatedCore, r1*r2*r3, true);//quantize surviving core elements and safe them

	//truncating the factor matrices
	Eigen::MatrixXd temp = us[0].leftCols(r1);
	us[0] = temp;
	temp = us[1].leftCols(r2);
	us[1] = temp;
	temp = us[2].leftCols(r3);
	us[2] = temp;

	//quantize the factor matrizes
	QuantizeData(us[0].data(), us[0].cols()*us[0].rows(),false);
	QuantizeData(us[1].data(), us[1].cols()*us[1].rows(), false);
	QuantizeData(us[2].data(), us[2].cols()*us[2].rows(), false);

}

void TensorTruncation::ReTruncateTensor(Eigen::Tensor<myTensorType, 3>& b, std::vector<Eigen::MatrixXd>& us, int d1, int d2, int d3)
{
	//reconstruct original-size Factor Matrizes
	Eigen::MatrixXd u1(d1, d1);
	u1.setZero();
	for (int x = 0; x < b.dimension(0); x++) {
		for (int y = 0;y < d1; y++) {
			u1(y, x) = us[0](y, x);
		}
	}
	us[0] = u1;

	Eigen::MatrixXd u2(d2, d2);
	u2.setZero();
	for (int x = 0; x < b.dimension(1); x++) {
		for (int y = 0;y < d2; y++) {
			u2(y, x) = us[1](y, x);
		}
	}
	us[1] = u2;

	Eigen::MatrixXd u3(d3, d3);
	u3.setZero();
	for (int x = 0; x < b.dimension(2); x++) {
		for (int y = 0;y < d3; y++) {
			u3(y, x) = us[2](y, x);
		}
	}
	us[2] = u3;

	//reconstruct original-size Tensor and fill it
	Eigen::Tensor<myTensorType, 3> ReconstructTensor(d1,d2,d3);
	ReconstructTensor.setZero();
	
	for (int z = 0;z < b.dimension(2); z++) {
		for (int x = 0;x < b.dimension(1); x++) {
			for (int y = 0;y < b.dimension(0); y++) {
				ReconstructTensor(y, x, z) = b(y, x, z);
			}
		}
	}

	b = ReconstructTensor;

}

//Truncate Core and then safe data with tthresh
void TensorTruncation::TruncateTensorTTHRESH(Eigen::Tensor<myTensorType, 3>& b, std::vector<Eigen::MatrixXd>& us, double errorTarget)
{
	//calculate core slice norms in order to calculate how much of the tensor we are allowed to cut off
	int zeroCols[3];
	//zeroCols[0] = 16;
	//zeroCols[1] = 56;
	//zeroCols[2] = 4;

	TensorOperations::calcCoreSliceNorms(b);

	for (int i = 0;i < TensorOperations::coreSliceNorms.size();i++) {
		for (int j = 0;j < TensorOperations::coreSliceNorms[i].size();j++) {
			us[i].col(j) *= TensorOperations::coreSliceNorms[i][j]; //scale columns of factor matrices with core slice norms in order to scale for overall error
		}

		double scale = 0;
		int p = TTHRESHEncoding::calcRLEP(us[i].data(), us[i].cols()*us[i].rows(), errorTarget, scale);//dummy run the rle to calculate stopping plane p and scale factor

		int countZeroCols = 0;
		for (int j = 0;j < us[i].cols();j++) {//calculate how many columns of the factor matrix would be safed as 0, these can be cut off without introducing an error!
			bool allZeroes = true;
			for (int k = 0;k < us[i].rows();k++) {
				if (abs(us[i](k,j)*scale) >= pow(2,p)) {//all coefficients that are < 2^p are cut off in the rle step
					allZeroes = false;
					break;
				}
			}
			if (allZeroes) {
				countZeroCols++;
			}
		}

		zeroCols[i] = countZeroCols;

	}

	std::cout << "Zeroes: " << zeroCols[0] << " " << zeroCols[1] << " "<< zeroCols[2] << std::endl;

	//calculate how much to cut off
	int r1 = b.dimension(0)- zeroCols[0];
	int r2 = b.dimension(1)- zeroCols[1];
	int r3 = b.dimension(2)- zeroCols[2];

	findRnAccError(r1,r2,r3, 0.01, b); //better solution: find Rn according to target error of AC step

	std::cout << "R: " << r1 << " " << r2 << " " << r3 << std::endl;

	//write characteristic data for decoding
	BitIO::writeBit(uint64_t(b.dimension(0)), 32);
	BitIO::writeBit(uint64_t(b.dimension(1)), 32);
	BitIO::writeBit(uint64_t(b.dimension(2)), 32);

	BitIO::writeBit(uint64_t(r1), 32);
	BitIO::writeBit(uint64_t(r2), 32);
	BitIO::writeBit(uint64_t(r3), 32);


	//truncate core and factor matrizes accordingly
	myTensorType* tCoreData = truncateTensorWithoutQuant(b, us, r1, r2, r3);

	//Save data with RLE+AC
	std::vector<std::vector<int>> rle;
	std::vector<std::vector<bool>> raw;
	std::vector<bool> signs;
	double scale = 0;

	std::vector<uint64_t> CoreMask = TTHRESHEncoding::encodeRLE(tCoreData, r1*r2*r3, errorTarget, true, rle, raw, scale, signs, b);//encode the core with rle+ac

	//encode the factor matrices: calculate core-slice norms TODO rballester special case 0
	Eigen::Tensor<myTensorType, 3> maskTensor = b; //TensorOperations::createTensorFromArray((myTensorType*)CoreMask.data(), b.dimension(0), b.dimension(1), b.dimension(2));//b
	std::vector<std::vector<double>> usNorms;

	for (int i = 0;i < us.size();i++) {//multiply each U col with core-slice norm
		std::vector<double> n;

		for (int j = 0;j < us[i].cols();j++) {
			//us[i].col(j) *= TensorOperations::coreSliceNorms[i][j];
			n.push_back(TensorOperations::coreSliceNorms[i][j]);
		}
		usNorms.push_back(n);
	}

	for (int i = 0;i < usNorms.size();i++) {//saving of norms for decompression
		BitIO::writeBit(usNorms[i].size(), 64);
		for (int j = 0;j < usNorms[i].size();j++) {
			uint64_t tmp;
			memcpy(&tmp, (void*)&usNorms[i][j], sizeof(usNorms[i][j]));
			BitIO::writeBit(tmp, 64);//write slicenorms to memory
		}
	}

	std::vector<std::vector<std::vector<int>>> usRle;
	std::vector<std::vector<std::vector<bool>>> usRaw;
	std::vector<double> usScales;
	std::vector<std::vector<bool>> usSigns;

	for (int i = 0; i < us.size(); i++) {//encode the factor matizes
		usRle.push_back(std::vector < std::vector<int>>());
		usRaw.push_back(std::vector < std::vector<bool>>());
		usScales.push_back(0);
		usSigns.push_back(std::vector<bool>());

		TTHRESHEncoding::encodeRLE(us[i].data(), us[i].cols()*us[i].rows(), errorTarget, false, usRle[i], usRaw[i], usScales[i], usSigns[i], b);//encode the factor matrices with rle+ac
	}

}

void TensorTruncation::RetruncateTensorTTHRESH(Eigen::Tensor<myTensorType, 3>& b, std::vector<Eigen::MatrixXd>& us, int d1, int d2, int d3, int r1, int r2, int r3)
{

	ReTruncateTensor(b, us, d1, d2, d3);//fill missing values with 0

}

myTensorType* TensorTruncation::truncateTensorWithoutQuant(Eigen::Tensor<myTensorType, 3>& b, std::vector<Eigen::MatrixXd>& us, int r1, int r2, int r3)
{
	myTensorType* truncatedCore = (myTensorType*)malloc(sizeof(myTensorType)*(r1*r2*r3)); //pointer to hold surviving core data

	//truncating the core
	int coreCounter = 0;
	for (int z = 0;z < r3; z++) {
		for (int x = 0;x < r2; x++) {
			for (int y = 0;y < r1; y++) {
				truncatedCore[coreCounter++] = b(y, x, z);
			}
		}
	}


	//truncating the factor matrices
	Eigen::MatrixXd temp = us[0].leftCols(r1);
	us[0] = temp;
	temp = us[1].leftCols(r2);
	us[1] = temp;
	temp = us[2].leftCols(r3);
	us[2] = temp;


	return truncatedCore;
}

void TensorTruncation::findRnAccError(int & r1, int & r2, int & r3, double errorTarget, Eigen::Tensor<myTensorType, 3>& b)
{
	//build summed area table for fast computation of frob norm: needed for relative error estimate
	Eigen::Tensor<myTensorType, 3> SAT(b.dimension(0), b.dimension(1), b.dimension(2));
	SAT.setZero();
	TensorTruncation::buildSummedAreaTable(b, SAT);

	//calculating all possibilities
	double frobNormSquared = SAT(b.dimension(0) - 1, b.dimension(1) - 1, b.dimension(2) - 1);
	double frobNorm = sqrt(frobNormSquared);
	double D123 = b.dimension(0)*b.dimension(1)*b.dimension(2);

	std::vector<OptimalChoice> choices;//data container to hold all choices for r1,r2,r3 and rel error, compression factor

	for (int r3 = 1; r3 <= b.dimension(2); r3++) {
		for (int r2 = 1; r2 <= b.dimension(1); r2++) {
			for (int r1 = 1; r1 <= b.dimension(0); r1++) {

				//formulas according to R.Ballester Paper "Lossy Volume Compression using Tucker truncation and thresholding"
				double F = ((r1*r2*r3) + (r1*b.dimension(0)) + (r2*b.dimension(1)) + (r3*b.dimension(2))) / D123;
				double relErr = sqrt(frobNormSquared - SAT(r1 - 1, r2 - 1, r3 - 1)) / frobNorm;

				choices.push_back(createChoiceNode(relErr, F, r1, r2, r3));

			}
		}
	}

	std::vector<OptimalChoice> C;//data container to hold all best choices for r1,r2,r3 and rel error, compression factor
	//sort choice-vector in increasing order (repsect to F)
	std::sort(choices.begin(), choices.end(), compareByF);
	double bestError = DBL_MAX;
	for (int i = 0;i < choices.size();i++) {

		if (choices[i].rE < bestError) {//found improvement
			bestError = choices[i].rE;
			C.push_back(choices[i]);
		}

	}

	//find the optimal r1, r2, r3 values so that error won't be noticeable
	for (int i = 0;i < C.size();i++) {//find best re-f pair
		if (C[i].rE <= errorTarget || i == C.size() - 1) {
			if ((i > 0) && abs(C[i].rE - errorTarget) > abs(C[i - 1].rE - errorTarget)) {
				r1 = C[i - 1].r1;
				r2 = C[i - 1].r2;
				r3 = C[i - 1].r3;
			}
			else {
				r1 = C[i].r1;
				r2 = C[i].r2;
				r3 = C[i].r3;
			}
			break;
		}
	}

}
