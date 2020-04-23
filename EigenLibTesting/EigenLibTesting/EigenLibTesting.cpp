// EigenLibTesting.cpp: Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#define _USE_MATH_DEFINES

#include <iostream>
#include <Eigen/Dense>
#include <math.h>

#include "VolInputParser.h"
#include "TensorOperations.h"
#include "TTHRESHEncoding.h"

using namespace Eigen;
using namespace std;

const int mysize = 8;

void createMatrix() {

	MatrixXd m(2, 2);
	m(0, 0) = 3;
	m(1, 0) = 2.5;
	m(0, 1) = -1;
	m(1, 1) = m(1, 0) + m(0, 1);
	std::cout << m << std::endl;

	Matrix3i testMat;

	testMat << 3, 2, 1,  //easy way to initialize
				1, 2, 3,
				2, 3, 1;

	cout << testMat << endl;

	//naming convention: datatype, size(square), membertype
	Matrix3d m1; // 3*3 double matix

	Matrix<float, 30, 10> m2; //30*10 float matrix

	int i = 10;
	MatrixXd m3(10,i); //compiletime unknown size

	//important for init: eigen stores matrices in COLUMN MAJOR order! also i j -> m(j,i)=0; initialisieren

	//special Matrices:
	Matrix4d id = Matrix4d::Identity(); //identity matrix

	Matrix4d z = Matrix4d::Zero(); //null matrix

	Matrix4d one = Matrix4d::Ones(); //ones matrix

	Matrix4d con = Matrix4d::Constant(3); //init all entries to a constant number

	//Arithmetic operations are overloaded to work with matrices

	//common matrix operations as member functions:

	cout << id.transpose() << endl;
	cout << con.inverse() << endl;

	//we can also access matrices element-wise with the array()-method

	cout << con.array().square() << endl;
	cout << con(1, 1) << endl;

}

void createVector() {

	Vector3d v1 (1, 2, 3); // init with << works too: v1 << 1, 2, 3;
	cout << v1 << endl;
	cout << "Second element: " << v1(2) << endl;

	//utility functions same as for matrices:
	cout << Vector3d::Ones() << endl;

	//arithmetic operations, scalar mult:
	cout << v1 + Vector3d(2, 3, 4) << endl;
	cout << v1 * 2 << endl;

	Vector3d v2(3, 4, 5);

	cout << v1 * v2.transpose() << endl; //caution for v*v: since they are matrices the inner dimensions have to match. Solution: Transpose function

	//linea algebra operations:
	cout << "Linalg: " << v1.dot(v2) << endl << v1.normalized() << endl << v1.cross(v2) << endl;

	VectorXd v3(3);
	v3 << 1, 2, 3;
	cout << v3.size() << endl;

}

void vertexTranformation(float * vertices) { //method that applies a transformation to a set of vertices

	MatrixXf mat = Map < Matrix<float, 3, mysize>>(vertices); //read matrix from float array TODO: dynamic size: use Xf matrix and read in from vector

	Transform<float, 3, Affine> trans = Transform<float, 3, Affine>::Identity();//init transfor matrix to hold our following transformations

	trans.scale(0.8); //scaling of vertices by 0.8
	trans.rotate( AngleAxisf(0.25f * M_PI,Vector3f::UnitX()));//rotating around x axis
	trans.translate(Vector3f(1.5, 10.2, -5.1));//translating in this vector

	cout << trans * mat.colwise().homogeneous() << endl; //apply transformations. Be careful of inner dimensions: homogenous(4x4), colwise

}

void unfoldTesting(VolInputParser dummy) {
	Eigen::MatrixXd ufold1 = TensorOperations::unfold(dummy.DummyTensor, 1);
	cout << endl << "Unfold 1:" << endl << ufold1 << endl;

	Tensor<myTensorType, 3> refold1 = TensorOperations::fold(ufold1, 1, 2, 3, 2);
	cout << endl << "Fold 1:" << endl << refold1 << endl;

	Eigen::MatrixXd ufold2 = TensorOperations::unfold(dummy.DummyTensor, 2);
	cout << endl << "Unfold 2:" << endl << ufold2 << endl;

	Tensor<myTensorType, 3> refold2 = TensorOperations::fold(ufold2, 2, 2, 3, 2);
	cout << endl << "Fold 2:" << endl << refold2 << endl;

	Eigen::MatrixXd ufold3 = TensorOperations::unfold(dummy.DummyTensor, 3);
	cout << endl << "Unfold 3:" << endl << ufold3 << endl;

	Tensor<myTensorType, 3> refold3 = TensorOperations::fold(ufold3, 3, 2, 3, 2);
	cout << endl << "Fold 3:" << endl << refold3 << endl;
}

void eigenTesting(VolInputParser dummy) {
	Eigen::MatrixXd dummyFold1 = TensorOperations::unfold(dummy.DummyTensor, 1);
	Eigen::MatrixXd dummy2(2, 2);
	dummy2 << 2, 2,
		1, 1;

	cout << endl << "Dummy2:" << endl << dummy2 << endl;
	cout << endl << "B*B^T: " << endl << dummy2 * dummy2.transpose() << endl;
	MatrixXd erg = TensorOperations::computeEigenvectors(dummy2);
	cout << endl << "U: " << endl << erg << endl;
	cout << endl << "Core: " << endl << erg.transpose()*dummy2 << endl;

	cout << endl << "Unfold 1:" << endl << dummyFold1 << endl;
	cout << endl << "B*B^T: " << endl << dummyFold1 * dummyFold1.transpose() << endl;
	MatrixXd erg2 = TensorOperations::computeEigenvectors(dummyFold1);
	cout << endl << "U: " << endl << erg2 << endl;
	cout << endl << "Core: " << endl << erg2.transpose()*dummyFold1 << endl;
	
	/*EigenSolver<MatrixXd> nSolver(dummy2*dummy2.transpose());
	Eigen::MatrixXd U2 = nSolver.eigenvectors().real();
	cout << endl << "U2: " << endl << U2 << endl;*/
}

void hosvdTesting(VolInputParser dummy) {
	cout << "starting hosvd" << endl;
	
	vector<MatrixXd> us(3);
	Tensor<myTensorType, 3> b;

	TensorOperations::HOSVD(dummy.DummyTensor, b, us);

	cout << endl << "Core: " << endl << b << endl;
	cout << endl << "U1: "<< endl << us[0] << endl;
	cout << endl << "U2: "<< endl << us[1] << endl;
	cout << endl << "U3: "<< endl << us[2] << endl;

	Eigen::Tensor<myTensorType, 3> decompressed;
	TensorOperations::decompress_HOSVD(b, decompressed, us);
	cout << endl << "Decompress: " << endl << decompressed<< endl << endl;
	
	vector<std::vector<int>> rle;
	vector<std::vector<int>> raw;
	cout << "starting encode: " << endl;
	TTHRESHEncoding::encodeRLE(b.data(), b.dimension(0)*b.dimension(1)*b.dimension(2),3, rle, raw);

}

void BitTesting(int num, int k) {
	uint64_t n[1];
	n[0] = num;
	
	unsigned * myBit = TTHRESHEncoding::getBits(n, k, 1);
	cout << "Bit " << k << ": " << myBit[0] << endl;
}

void RLETesting() {

	uint64_t n[8];
	n[0] = 0;
	n[1] = 1;
	n[2] = 1;
	n[3] = 1;
	n[4] = 0;
	n[5] = 0;
	n[6] = 0;
	n[7] = 1;


	unsigned * bits = TTHRESHEncoding::getBits(n, 0, 8);

	vector<int> rle = TTHRESHEncoding::RLE(bits, 8);

	for (int i = 0; i < rle.size();i++) {
		cout << rle[i];
	}
	cout << endl;
}
 
int main()
{

	cout << "0 for matrix testing, 1 for vector testing" << endl << "Eingabe: ";
	int input;
	cin >> input;

	float arrVertices[] = { -1.0 , -1.0 , -1.0 ,
		1.0 , -1.0 , -1.0 ,
		1.0 , 1.0 , -1.0 ,
		-1.0 , 1.0 , -1.0 ,
		-1.0 , -1.0 , 1.0 ,
		1.0 , -1.0 , 1.0 ,
		1.0 , 1.0 , 1.0 ,
		-1.0 , 1.0 , 1.0 };

	//char hehe[] = "stagbeetle277x277x164.dat";//"stagbeetle832x832x494.dat";
	//VolInputParser parse = VolInputParser(hehe);
	//std::cout << "Debug Data Test: " << parse.TensorData(12, 101, 6) << std::endl;

	VolInputParser dummy = VolInputParser();
	//unfoldTesting(dummy);

	//eigenTesting(dummy);

	hosvdTesting(dummy);
	//TensorOperations::getSlice(dummy.DummyTensor,1,0);
	//RLETesting();

	switch (input)
	{
	case 1:
		createVector();
		break;

	case 2:
		vertexTranformation(arrVertices);
		break;
		
	case 3:
		break;

	default:
		createMatrix();
		break;
	}

	return 0;
}
