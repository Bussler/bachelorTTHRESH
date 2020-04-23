#include "TensorOperations.h"



TensorOperations::TensorOperations()
{
}


TensorOperations::~TensorOperations()
{
}

//function that takes Matrix input and unfoldes it into given mode TODO:Maybe better with chip, see getSlice()
Eigen::MatrixXd TensorOperations::unfold(Eigen::Tensor<myTensorType, 3> input, int mode)
{
	int dimY = input.dimension(mode-1);
	int dimX = 0;
	Eigen::MatrixXd mat;

	switch (mode) {
	case 1:
		dimX = input.dimension(1)*input.dimension(2);
		mat = Eigen::MatrixXd(dimY, dimX);//(1,0) -> x wert 0, y wert 1
		for (int x = 0; x < dimX; x++) {
			for (int y = 0; y < dimY; y++) {
				mat(y, x) =  input(y, x%input.dimension(1), x / input.dimension(1));
			}
		}

		break;

	case 2:
		dimX = input.dimension(0)*input.dimension(2);
		mat = Eigen::MatrixXd(dimY, dimX);
		for (int x = 0; x < dimX; x++) {
			for (int y = 0; y < dimY; y++) {
				mat(y, x) = input(x%input.dimension(2), y, x / input.dimension(2));
			}
		}
		break;

	case 3:
		dimX = input.dimension(0)*input.dimension(1);
		mat = Eigen::MatrixXd(dimY, dimX);
		for (int x = 0; x < dimX; x++) {
			for (int y = 0; y < dimY; y++) {
				mat(y, x) = input(x%input.dimension(0), x/input.dimension(0), y);
			}
		}
		break;
	}

	return mat;
}

Eigen::Tensor<myTensorType, 3> TensorOperations::fold(Eigen::MatrixXd input, int mode, int x, int y, int z)
{
	int dimY = input.rows();
	int dimX = input.cols();

	Eigen::Tensor<myTensorType, 3> tensor(x, y, z);

	switch (mode) {
	case 1:
		for (int x = 0; x < dimX; x++) {
			for (int y = 0; y < dimY; y++) {
				tensor(y, x%tensor.dimension(1), x/tensor.dimension(1)) = input(y, x);
			}
		}

		break;

	case 2:
		for (int x = 0; x < dimX; x++) {
			for (int y = 0; y < dimY; y++) {
				tensor(x%tensor.dimension(2),y,x/tensor.dimension(2)) = input(y, x);
			}
		}
		break;

	case 3:
		for (int x = 0; x < dimX; x++) {
			for (int y = 0; y < dimY; y++) {
				tensor(x%tensor.dimension(0), x/tensor.dimension(0), y) = input(y, x);
			}
		}
		break;
	}


	return tensor;
}

//function that computes Factormatrix U for unfolded Core B
Eigen::MatrixXd TensorOperations::computeEigenvectors(Eigen::MatrixXd B)
{
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(B*B.transpose()); //B*B^T is symmetric, we are able to use faster Eigen comp

	Eigen::MatrixXd U_unsorted = solver.eigenvectors().real();//computing the eigenvectors for U, but still unordered
	int rows = U_unsorted.rows();
	
	Eigen::MatrixXd U(rows, rows);// Factor Matrix, eigenvectors are cols

	//Order eigenvectors column wise, descending eigenval
	Eigen::VectorXd eigVal = solver.eigenvalues().real();

	std::vector< std::pair<myTensorType, int>> sortedEigenval(rows);//constant size vector, this way I don't have to handle pointer-array
	for (int i = 0; i < rows; i++) {//read in eigenvalues from top to bottom
		sortedEigenval[i] = std::pair<myTensorType, int>((-1)*eigVal(i), i); // read in neg values for desc sorting
	}

	std::sort(sortedEigenval.begin(), sortedEigenval.end());//sort eigenvalue pairs desc

	for (int i = 0;i < rows;i++) {
		U.col(i) = U_unsorted.col(sortedEigenval[i].second);//rearrange cols accoring to sorting
	}

	return U;
}


//TODO: way to compute without folding into tensor: matrix -> matrix
void TensorOperations::HOSVD(Eigen::Tensor<myTensorType, 3> T, Eigen::Tensor<myTensorType, 3>& B, std::vector<Eigen::MatrixXd>& Us)
{
	B = T;
	int x = T.dimension(0); int y = T.dimension(1); int z = T.dimension(2);

	int n = T.NumDimensions;

	for (int i = 1; i <= n; i++) {//for n modes
		Eigen::MatrixXd unfoldedB = unfold(B, i);//unfold tensor into 2d matrix

		Eigen::MatrixXd Un = computeEigenvectors(unfoldedB);//compute FactorMatrix U for unfolded core

		unfoldedB = Un.transpose()*unfoldedB;//calculate new Core with E*V^T = Un^T*Bn

		B = fold(unfoldedB, i, x, y, z);//fold back 2d core into tensor
		Us[i - 1] = Un;//store the factor matrix Un
	}

}

//TTM: contract n-mode fibres along U matrix rows and fold back into tensor
void TensorOperations::TTM(Eigen::Tensor<myTensorType, 3> &T, Eigen::MatrixXd U, int mode)
{
	Eigen::MatrixXd ufold= unfold(T, mode);

	for (int i = 0; i < ufold.cols(); i++) {
		ufold.col(i) = U * ufold.col(i);
	}

	T = fold(ufold, mode, T.dimension(0), T.dimension(1), T.dimension(2));

}


//Calculate original Tensor according to schema T=B x1 U1 ... xn Un
void TensorOperations::decompress_HOSVD(Eigen::Tensor<myTensorType, 3> B, Eigen::Tensor<myTensorType, 3>& T, std::vector<Eigen::MatrixXd> Us)
{
	T = B;
	int n = Us.size();

	for (int i = 0;i < n; i++) {
		
		TTM(T, Us[i], i + 1);

	}
}

//calculates i-th slice of tensor in dimension dim
Eigen::MatrixXd TensorOperations::getSlice(Eigen::Tensor<myTensorType, 3> T, int dim, int i)
{
	Eigen::MatrixXd slice;
	Eigen::Tensor<myTensorType, 2> sl;
	/*
	switch (dim)
	{
	case 1:
		sl = T.chip(i, 2);
		slice = Eigen::Map<const Eigen::MatrixXd>(sl.data(), T.dimension(0), T.dimension(1));//transform tensor into matrix
		break;

	case 2://TODO vlt spiegeln
		sl = T.chip(i, 1);
		slice = Eigen::Map<const Eigen::MatrixXd>(sl.data(), T.dimension(0), T.dimension(2));//transform tensor into matrix
		break;

	case 3:
		sl = T.chip(i, 0);
		slice = Eigen::Map<const Eigen::MatrixXd>(sl.data(), T.dimension(1), T.dimension(2));//transform tensor into matrix
		break;
	}
	std::cout << std::endl << "Slice " << i << " mode " << dim << " : " << std::endl << slice << std::endl;
	*/
	return slice;
}

/*template<typename Scalar, int rank, typename sizeType>
auto Tensor_to_Matrix(const Eigen::Tensor<Scalar, rank> tensor, const sizeType rows, const sizeType cols)
{
	return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), rows, cols);
}*/
