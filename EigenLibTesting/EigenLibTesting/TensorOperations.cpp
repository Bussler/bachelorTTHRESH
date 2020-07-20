#include "TensorOperations.h"

std::vector< std::vector < double>> TensorOperations::coreSliceNorms;

//function that takes Matrix input and unfoldes it into given mode TODO:Maybe better with chip, see getSlice()
Eigen::MatrixXd TensorOperations::unfold(Eigen::Tensor<myTensorType, 3>& input, int mode)
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
				mat(y, x) = input(x / input.dimension(2), y, x % input.dimension(2));
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



Eigen::Tensor<myTensorType, 3> TensorOperations::fold(Eigen::MatrixXd& input, int mode, int x, int y, int z)
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
				tensor(x/tensor.dimension(2), y, x % tensor.dimension(2)) = input(y, x);
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

//TODO welp
Eigen::MatrixXd TensorOperations::unfoldTensor(Eigen::Tensor<myTensorType, 3> input, int mode)
{
	int dimY = input.dimension(mode - 1);
	int dimX = 0;

	Eigen::MatrixXd mat;
	Eigen::MatrixXd mBlock;
	int c = 1;
	
	switch (mode)
	{
	case 1:
		dimX = input.dimension(1)*input.dimension(2);
		mat = Eigen::Map<Eigen::MatrixXd>(input.data(), dimY, dimX);

		//mBlock = mat.block(0, 0, 2, 3);
		//std::cout << "Block: " << std::endl << mBlock << std::endl;

		break;

	case 2:
		dimX = input.dimension(0)*input.dimension(2);

		mat = Eigen::Map<Eigen::MatrixXd,0, Eigen::Stride<0,Eigen::Dynamic>>(input.data(), dimY, dimX, Eigen::Stride<0, Eigen::Dynamic>(0, input.dimension(0))); // <,2>
		break;

	case 3:
		dimX = input.dimension(0)*input.dimension(1);
		mat = Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(input.data(), dimY, dimX, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, input.dimension(0)*input.dimension(1)));
		break;
	}

	std::cout <<"unfolded dim "<<mode<< ": "<<std::endl<< mat << std::endl;

	//Eigen::MatrixXd C(mat.rows(), mat.cols()+mat.cols());
	//C << mat, mat;

	return mat;
}

//function that computes Factormatrix U for unfolded Core B 
Eigen::MatrixXd TensorOperations::computeEigenvectors(Eigen::MatrixXd& B)
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
void TensorOperations::HOSVD(Eigen::Tensor<myTensorType, 3>& T, Eigen::Tensor<myTensorType, 3>& B, std::vector<Eigen::MatrixXd>& Us)
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
void TensorOperations::TTM(Eigen::Tensor<myTensorType, 3> &T, Eigen::MatrixXd& U, int mode)
{
	Eigen::MatrixXd ufold= unfold(T, mode);

	for (int i = 0; i < ufold.cols(); i++) {
		ufold.col(i) = U * ufold.col(i);
	}

	T = fold(ufold, mode, T.dimension(0), T.dimension(1), T.dimension(2));

}


//Calculate original Tensor according to schema T=B x1 U1 ... xn Un
void TensorOperations::decompress_HOSVD(Eigen::Tensor<myTensorType, 3>& B, Eigen::Tensor<myTensorType, 3>& T, std::vector<Eigen::MatrixXd>& Us)
{
	T = B;
	int n = Us.size();

	for (int i = 0;i < n; i++) {
		
		TTM(T, Us[i], i + 1);

	}
}

//convert array back to Eigen::Tensor Type for further use (Core Tensor)
Eigen::Tensor<myTensorType, 3> TensorOperations::createTensorFromArray(myTensorType * data, int d1, int d2, int d3)
{
	Eigen::Tensor<myTensorType, 3> t = Eigen::TensorMap<Eigen::Tensor<myTensorType, 3>>(data, d1, d2, d3);
	return t;
}

//convert array back to Eigen::Matrix Type for further use (Factor Matrizes)
Eigen::MatrixXd TensorOperations::createMatrixFromArray(double* data, int d1, int d2)
{
	Eigen::MatrixXd m = Eigen::Map<Eigen::MatrixXd>(data, d1, d2);
	return m;
}



//calculates i-th slice of tensor in dimension dim TODO slices correct this way?
Eigen::MatrixXd TensorOperations::getSlice(Eigen::Tensor<myTensorType, 3>& T, int dim, int i)
{
	Eigen::MatrixXd slice;
	Eigen::Tensor<myTensorType, 2> sl;


	switch (dim)
	{
	case 1:
		sl = T.chip(i, 2);
		slice = Eigen::Map<const Eigen::MatrixXd>(sl.data(), T.dimension(0), T.dimension(1));//transform tensor into matrix 
		break;

	case 2:
		sl = T.chip(i, 1);
		slice = Eigen::Map<const Eigen::MatrixXd>(sl.data(), T.dimension(0), T.dimension(2));//transform tensor into matrix 
		break;

	case 3:
		sl = T.chip(i, 0);
		slice = Eigen::Map<const Eigen::MatrixXd>(sl.data(), T.dimension(1), T.dimension(2));//transform tensor into matrix
		break;
	}
	
	return slice;
}

//Function to reorder the core coefficients, so that the hot corner elemnts come first
double * TensorOperations::reorderCore(Eigen::Tensor<myTensorType, 3>& B)
{
	int dim1 = B.dimension(0);
	int dim2 = B.dimension(1);
	int dim3 = B.dimension(2);

	double * orderedData = (double*)malloc(sizeof(double)*(dim1*dim2*dim3));
	int counter = 0;

	for (int mhD = 0;mhD < dim1 + dim2 + dim3 -2;mhD++) {//manhattan distance: order elements with smallest manhatten distance to hot corner (0,0,0) first
		//std::cout << "Distance: " << mhD << std::endl;
		int y = 0;
		int x = 0;
		int z = 0;

		do
		{
			x = 0;
			do
			{
				if (mhD - y - x < dim3) {
					z = mhD - y - x;
					orderedData[counter++] = B(y,x,z);
					//std::cout << "(" << y << " , " << x << " , " << z << ")" << std::endl;
				}

				x++;
			} while (x <= mhD - y && x < dim2);
			 
			y++;
		} while (y <= mhD && y < dim1);

	}

	return orderedData;
}

double * TensorOperations::reorderCoreMajor(Eigen::Tensor<myTensorType, 3>& b)
{
	//column major ordering yzx
	Eigen::Tensor<myTensorType, 3> testo(b.dimension(0), b.dimension(2), b.dimension(1));
	for (int i = 0;i < testo.dimension(2);i++) {
		for (int j = 0;j < testo.dimension(1);j++) {
			for (int k = 0;k < testo.dimension(0);k++) {
				testo(k, j, i) = b(k, i, j);
			}
		}
	}


	//Row major reordering xyz
	//Eigen::Tensor<myTensorType, 3> testo(b.dimension(1), b.dimension(0), b.dimension(2));
	for (int i = 0;i < testo.dimension(2);i++) {
		for (int j = 0;j < testo.dimension(1);j++) {
			for (int k = 0;k < testo.dimension(0);k++) {
				testo(k, j, i) = b(j, k, i);
			}
		}
	}

	// xzy
	//Eigen::Tensor<myTensorType, 3> testo(b.dimension(1), b.dimension(2), b.dimension(0));
	for (int i = 0;i < testo.dimension(2);i++) {
		for (int j = 0;j < testo.dimension(1);j++) {
			for (int k = 0;k < testo.dimension(0);k++) {
				testo(k, j, i) = b(i, k, j);
			}
		}
	}

	//z major reordering zyx
	//Eigen::Tensor<myTensorType, 3> testo(b.dimension(2), b.dimension(0), b.dimension(1));
	for (int i = 0;i < testo.dimension(2);i++) {
		for (int j = 0;j < testo.dimension(1);j++) {
			for (int k = 0;k < testo.dimension(0);k++) {
				testo(k, j, i) = b(j, i, k);
			}
		}
	}

	// zxy
	//Eigen::Tensor<myTensorType, 3> testo(b.dimension(2), b.dimension(1), b.dimension(0));
	for (int i = 0;i < testo.dimension(2);i++) {
		for (int j = 0;j < testo.dimension(1);j++) {
			for (int k = 0;k < testo.dimension(0);k++) {
				testo(k, j, i) = b(i, j, k);
			}
		}
	}

	return testo.data();
}

double * TensorOperations::reorderCoreBtf(Eigen::Tensor<myTensorType, 3>& B)
{
	int dim1 = B.dimension(0);
	int dim2 = B.dimension(1);
	int dim3 = B.dimension(2);

	double * orderedData = (double*)malloc(sizeof(double)*(dim1*dim2*dim3));
	int counter = 0;

	for (int mhD = dim1 + dim2 + dim3 - 2; mhD >= 0; mhD--) {//manhattan distance: order elements with largest manhatten distance to hot corner (0,0,0) first
		//std::cout << "Distance: " << mhD << std::endl;
		int y = dim1-1;
		int x = dim2-1;
		int z = dim3-1;

		do
		{
			x = dim2-1;
			do
			{
				if (mhD - y - x < dim3 && mhD-y-x >= 0) {
					z = mhD - y - x;
					orderedData[counter++] = B(y, x, z);
					//std::cout << "(" << y << " , " << x << " , " << z << ")" << std::endl;
				}

				x--;
			} while (x >= 0);

			y--;
		} while (y >= 0);

	}

	return orderedData;
}

double * TensorOperations::reorderCoreWeighted(Eigen::Tensor<myTensorType, 3>& B, int weight, int dim)
{
	int dim1 = B.dimension(0);
	int dim2 = B.dimension(1);
	int dim3 = B.dimension(2);

	double * orderedData = (double*)malloc(sizeof(double)*(dim1*dim2*dim3));
	int counter = 0;
	int mhD = 0;


	int weightY = 1;
	int weightX = 2;
	int weightZ = 3;


	switch (dim)
	{
	case 1:

		while (counter < dim1*dim2*dim3) {
			//std::cout << "Distance: " << mhD << std::endl;
			int z = 0;
			int x = 0;
			int y = 0;

			do
			{
				x = 0;
				do
				{
					y = (mhD - z - x)/weight;
					if (y < dim1 && y >= 0 && (y*weight)+x+z==mhD) {
						orderedData[counter++] = B(y, x, z); //B(second, third, first);
						//std::cout << "(" << y << " , " << x << " , " << z << ")" << std::endl;
					}

					x++;
				} while (x <= mhD - z && x < dim2);

				z++;
			} while (z <= mhD && z < dim3);

			mhD++;
		}

		break;

	case 2:

		while (counter <dim1*dim2*dim3) {
			//std::cout << "Distance: " << mhD << std::endl;
			int z = 0;
			int x = 0;
			int y = 0;

			do
			{
				x = 0;
				do
				{
					if (mhD - z - (x*weight) < dim1 && mhD - z - (x*weight)>=0) {
						y = mhD - z - (x*weight);
						orderedData[counter++] = B(y, x, z); //B(second, third, first); z, x, y
						//std::cout << "(" << y << " , " << x << " , " << z << ")" << std::endl;
					}

					x++;
				} while (x <= mhD - z && x < dim2);

				z++;
			} while (z <= mhD && z < dim3);

			mhD++;
		}
		
		break;

	case 3:

		while (counter < dim1*dim2*dim3) {
			//std::cout << "Distance: " << mhD << std::endl;
			int z = 0;
			int x = 0;
			int y = 0;

			do
			{
				x = 0;
				do
				{
					if (mhD - (z*weight) - x < dim1 && mhD - (z*weight) - x >= 0) {
						y = mhD - (z*weight) - x;
						orderedData[counter++] = B(y, x, z); //B(second, third, first);
						//std::cout << "(" << y << " , " << x << " , " << z << ")" << std::endl;
					}

					x++;
				} while (x <= mhD - z && x < dim2);

				z++;
			} while (z <= mhD && z < dim3);

			mhD++;
		}

		break;

	case 4://Special Case: Weight multiple dim

		weightX = 164;
		weightY = 2;
		weightZ = 164;


		while (counter < dim1*dim2*dim3) {
			//std::cout << "Distance: " << mhD << std::endl;
			int z = 0;
			int x = 0;
			int y = 0;

			do
			{
				x = 0;
				do
				{

					y = (mhD - (z*weightZ) - (x*weightX)) / weightY;
					if (y < dim3 && y >= 0 && (y*weightY) + (x*weightX) + (z*weightZ) == mhD) {
						orderedData[counter++] = B(z, x, y); //B(second, third, first); (y, x, z)
						//std::cout << "(" << y << " , " << x << " , " << z << ")" << std::endl;
					}

					x++;
				} while (x <= mhD - z && x < dim2);

				z++;
			} while (z <= mhD && z < dim1);

			mhD++;
		}


		break;

	default:
		break;
	}

	return orderedData;
}


Eigen::Tensor<myTensorType, 3> TensorOperations::deorderCore(double * data, int d1, int d2, int d3)
{
	Eigen::Tensor<myTensorType, 3> B(d1, d2, d3);
	int counter = 0;

	for (int mhD = 0;mhD < d1 + d2 + d3 - 2;mhD++) {//maanhattan distance: order elements with smallest manhatten distance to hot corner (0,0,0) first

		int y = 0;
		int x = 0;
		int z = 0;

		do
		{
			x = 0;
			do
			{
				if (mhD - y - x < d3) {
					z = mhD - y - x;
					B(y, x, z) = data[counter++];
				}

				x++;
			} while (x <= mhD - y && x < d2);

			y++;
		} while (y <= mhD && y < d1);

	}

	return B;
}

//Calculates scale-Factors (Core Slice Norms) for Factor-Matrices by computing the eigenvalues. Do not use, high comp. cost
void TensorOperations::calcCoreSliceNormsFromEigenvalue(Eigen::Tensor<myTensorType, 3> B)
{
	for (int i = 1;i <= 3;i++) {

		Eigen::MatrixXd unfoldedB = unfold(B, i);//unfold tensor into 2d matrix

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(unfoldedB*unfoldedB.transpose()); //B*B^T is symmetric, we are able to use faster Eigen comp

		//Order eigenvectors column wise, descending eigenval
		Eigen::VectorXd eigVal = solver.eigenvalues().real();
		//std::cout <<"Eigenvalues: "<< eigVal << std::endl;
		std::vector< std::pair<myTensorType, int>> sortedEigenval(eigVal.rows());//constant size vector, this way I don't have to handle pointer-array
		for (int i = 0; i < eigVal.rows(); i++) {//read in eigenvalues from top to bottom
			sortedEigenval[i] = std::pair<myTensorType, int>((-1)*eigVal(i), i); // read in neg values for desc sorting
		}

		std::sort(sortedEigenval.begin(), sortedEigenval.end());//sort eigenvalue pairs desc

		std::vector<double> slices;
		for (int i = 0;i < eigVal.rows();i++) {
			slices.push_back((-1)*sortedEigenval[i].first); //norm of slice = eigval
		}
		coreSliceNorms.push_back(slices);

	}

}


void TensorOperations::calcCoreSliceNorms(Eigen::Tensor<myTensorType, 3> B)
{
	coreSliceNorms = std::vector< std::vector < double>>(3);

	for (int i = 0; i < 3; i++) {
		std::vector<double> n; //temp vector to store the slice norms

		int converted = 0;
		switch (i)
		{
		case 0: converted = 3;
			break;

		case 1: converted = 2;
			break;

		case 2: converted = 1;
			break;
		}

		for (int j = 0;j < B.dimension(i);j++) {
			Eigen::MatrixXd slice = TensorOperations::getSlice(B, converted, j);
			n.push_back(slice.norm()); //TensorOperations::coreSliceNorms[i][j]);
		}
		coreSliceNorms[i] = n;//saving in global vector
	}

}
