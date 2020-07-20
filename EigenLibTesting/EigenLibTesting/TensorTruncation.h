#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <map>
#include "VolInputParser.h"
#include "TensorOperations.h"


namespace TensorTruncation {

	struct OptimalChoice
	{
		double rE;//relative Error
		double F;//compression factor
		int r1;
		int r2;
		int r3;

	};

	int logQuantize(double val, double max);
	double logDequantize(int val, double max);
	void QuantizeData(double * coefficients, int numC, bool isCore);
	
	OptimalChoice createChoiceNode(double rE, double F, int r1, int r2, int r3);

	void buildSummedAreaTable(Eigen::Tensor<myTensorType, 3>& b, Eigen::Tensor<myTensorType, 3>& SAT);
	void CalculateTruncation(Eigen::Tensor<myTensorType, 3>& b, std::vector<Eigen::MatrixXd>& us, double givenRe);
	void CalculateRetruncation(Eigen::Tensor<myTensorType, 3>& b, std::vector<Eigen::MatrixXd>& us, int d1, int d2, int d3, int r1, int r2, int r3);

	void truncateTensor(Eigen::Tensor<myTensorType, 3>& b, std::vector<Eigen::MatrixXd>& us, int r1, int r2, int r3);
	void ReTruncateTensor(Eigen::Tensor<myTensorType, 3>& b, std::vector<Eigen::MatrixXd>& us, int d1, int d2, int d3);

	void TruncateTensorTTHRESH(Eigen::Tensor<myTensorType, 3>& b, std::vector<Eigen::MatrixXd>& us, double errorTarget);
	void RetruncateTensorTTHRESH(Eigen::Tensor<myTensorType, 3>& b, std::vector<Eigen::MatrixXd>& us, int d1, int d2, int d3, int r1, int r2, int r3);
	myTensorType* truncateTensorWithoutQuant(Eigen::Tensor<myTensorType, 3>& b, std::vector<Eigen::MatrixXd>& us, int r1, int r2, int r3);
	void findRnAccError(int & r1, int & r2, int & r3, double errorTarget, Eigen::Tensor<myTensorType, 3>& b);

}

