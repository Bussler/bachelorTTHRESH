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
void TensorTruncation::QuantizeData(double * coefficients, int numC) // TODO encode first element seperately!
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
	std::cout << std::endl << "U3: " << std::endl << us[2] << std::endl;*/

}
