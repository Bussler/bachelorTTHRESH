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
void TensorTruncation::QuantizeData(double * coefficients, int numC)
{
	//first find abs value of max element
	double max = 0;
	for (int i = 0;i < numC;i++) {
		if (abs(coefficients[i]) > max) {
			max = abs(coefficients[i]);
		}
	}

	double hotCornerElement = coefficients[0];//safe hot corner seperately, since it holds most of the core's energy
	//TODO safe hot corner element

	for (int i = 1;i < numC; i++) {
		int quant = logQuantize(coefficients[i], max);
		//TODO write Data in 9 bit here
		std::cout << "Orig: " << coefficients[i] << " Quant: " << quant << " Reconstructed: "<<logDequantize(quant,max) << std::endl;
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


void TensorTruncation::CalculateTruncation(Eigen::Tensor<myTensorType, 3>& b, std::vector<Eigen::MatrixXd>& us)
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

	//TODO test for (2,2,2)
	truncateTensor(b, us, 2, 2, 2);

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

	QuantizeData(truncatedCore, r1*r2*r3);//quantize surviving core elements and safe them

	//truncating the factor matrices
	us[0] = us[0].leftCols(r1);
	us[1] = us[1].leftCols(r2);
	us[2] = us[2].leftCols(r3);

	//TODO quantize the factor matrizes

	
	/*//DEBUGGING
	Eigen::Tensor<myTensorType,3> truncatedC = TensorOperations::createTensorFromArray(truncatedCore, r1, r2, r3);
	std::cout << "TRUNCATED DATA" << std::endl;
	std::cout << std::endl << "Core: " << std::endl << truncatedC << std::endl;
	std::cout << std::endl << "U1: " << std::endl << us[0] << std::endl;
	std::cout << std::endl << "U2: " << std::endl << us[1] << std::endl;
	std::cout << std::endl << "U3: " << std::endl << us[2] << std::endl;
	
	ReTruncateTensor(truncatedC, us, b.dimension(0), b.dimension(1), b.dimension(2));
	std::cout << "Retruncated: " << std::endl << truncatedC << std::endl;
	std::cout << std::endl << "U1: " << std::endl << us[0] << std::endl;
	std::cout << std::endl << "U2: " << std::endl << us[1] << std::endl;
	std::cout << std::endl << "U3: " << std::endl << us[2] << std::endl;*/

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
