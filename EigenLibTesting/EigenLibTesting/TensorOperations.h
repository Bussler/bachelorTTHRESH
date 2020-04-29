#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

typedef double myTensorType;

class TensorOperations
{
public:
	TensorOperations();
	~TensorOperations();

	static Eigen::MatrixXd unfold(Eigen::Tensor<myTensorType, 3> input, int mode);
	static Eigen::Tensor<myTensorType,3> fold(Eigen::MatrixXd input, int mode, int x, int y, int z);

	static Eigen::MatrixXd computeEigenvectors(Eigen::MatrixXd B);

	static void HOSVD(Eigen::Tensor<myTensorType, 3> T, Eigen::Tensor<myTensorType, 3>& B, std::vector<Eigen::MatrixXd>& Us);

	static void TTM(Eigen::Tensor<myTensorType, 3>& T, Eigen::MatrixXd U, int mode);

	static void decompress_HOSVD(Eigen::Tensor<myTensorType, 3> B, Eigen::Tensor<myTensorType, 3>& T, std::vector<Eigen::MatrixXd> Us);

	static Eigen::Tensor<myTensorType, 3> createTensorFromArray(myTensorType* data, int d1, int d2, int d3);
	static Eigen::MatrixXd createMatrixFromArray(double* data, int d1, int d2);

	static Eigen::MatrixXd getSlice(Eigen::Tensor<myTensorType, 3> T, int dim, int i);
};