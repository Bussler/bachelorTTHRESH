#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

typedef double myTensorType;

namespace TensorOperations
{
	extern std::vector< std::vector < double>>coreSliceNorms;

	Eigen::MatrixXd unfold(Eigen::Tensor<myTensorType, 3>& input, int mode);
	Eigen::Tensor<myTensorType,3> fold(Eigen::MatrixXd & input, int mode, int x, int y, int z);

	Eigen::MatrixXd unfoldTensor(Eigen::Tensor<myTensorType, 3> input, int mode);

	Eigen::MatrixXd computeEigenvectors(Eigen::MatrixXd& B);

	void HOSVD(Eigen::Tensor<myTensorType, 3>& T, Eigen::Tensor<myTensorType, 3>& B, std::vector<Eigen::MatrixXd>& Us);

	void TTM(Eigen::Tensor<myTensorType, 3>& T, Eigen::MatrixXd & U, int mode);

	void decompress_HOSVD(Eigen::Tensor<myTensorType, 3>& B, Eigen::Tensor<myTensorType, 3>& T, std::vector<Eigen::MatrixXd>& Us);

	Eigen::Tensor<myTensorType, 3> createTensorFromArray(myTensorType* data, int d1, int d2, int d3);
	Eigen::MatrixXd createMatrixFromArray(double* data, int d1, int d2);

	Eigen::MatrixXd getSlice(Eigen::Tensor<myTensorType, 3>& T, int dim, int i);
	double * reorderCore(Eigen::Tensor<myTensorType, 3>& B);
	double * reorderCore2(Eigen::Tensor<myTensorType, 3>& B);
	Eigen::Tensor<myTensorType, 3> deorderCore(double * data, int d1, int d2, int d3);

	void calcCoreSliceNorms(Eigen::Tensor<myTensorType, 3> B);
}