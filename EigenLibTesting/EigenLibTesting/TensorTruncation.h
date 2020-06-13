#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <map>
#include "VolInputParser.h"
#include "TensorOperations.h"


namespace TensorTruncation {

	int logQuantize(double val, double max);
	double logDequantize(int val, double max);
	void QuantizeData(double * coefficients, int numC);

	void truncateTensor(Eigen::Tensor<myTensorType, 3>& b, std::vector<Eigen::MatrixXd>& us, int r1, int r2, int r3);

}

