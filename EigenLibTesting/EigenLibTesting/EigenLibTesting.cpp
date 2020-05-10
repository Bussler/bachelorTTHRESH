// EigenLibTesting.cpp: Definiert den Einstiegspunkt für die Konsolenanwendung.
//
#pragma warning(disable : 4996)
#define _USE_MATH_DEFINES

#include <iostream>
#include <Eigen/Dense>
#include <math.h>

#include "VolInputParser.h"
#include "TensorOperations.h"
#include "TTHRESHEncoding.h"
#include <fstream>

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
	vector<std::vector<bool>> raw;
	vector<bool> signs;
	double scale = 0;
	cout << "starting encode: " << endl;

	dummy.writeCharacteristicData(b.dimension(0), b.dimension(1), b.dimension(2), us[0].rows(), us[0].cols(), us[1].rows(), us[1].cols(), us[2].rows(), us[2].cols());

	TTHRESHEncoding::compress(b, us, 0.0003, TTHRESHEncoding::ErrorType::epsilon, rle, raw, scale, signs);

	//cout << "OrigScale: " << scale << endl;
	BitIO::writeRemainingBit();
	BitIO::closeWrite();


	//AC encoding of rle data
	char acTxt[] = "AC.txt";
	BitIO::openWrite(acTxt); //Open document to write into
	TTHRESHEncoding::encodeACVektor(rle);
	BitIO::writeRemainingBit();
	BitIO::closeWrite();

	BitIO::openRead(acTxt);
	vector<vector<int>> decRleB;
	TTHRESHEncoding::decodeACVektor(decRleB);
	BitIO::closeRead();

	/*
	char acTxt[] = "AC.txt";
	BitIO::openWrite(acTxt); //Open document to write into

	for (int i = 0;i < rle.size();i++) {
		TTHRESHEncoding::encodeAC(rle[i]);
	}

	BitIO::writeRemainingBit();

	BitIO::closeWrite();

	//AC decoding of rl data
	BitIO::openRead(acTxt);
	vector<vector<int>> decRleB;
	for (int i = 0;i < rle.size();i++) {
		vector<int> dRle;
		TTHRESHEncoding::decodeAC(dRle);
		decRleB.push_back(dRle);
	}
	BitIO::closeRead();*/

	//double * dec = TTHRESHEncoding::decodeRLE(rle, raw, b.dimension(0)*b.dimension(1)*b.dimension(2), scale, signs);
	double * dec = TTHRESHEncoding::decodeRLE(decRleB, raw, b.dimension(0)*b.dimension(1)*b.dimension(2), scale, signs);

	Eigen::Tensor<myTensorType, 3> decompressedB = TensorOperations::createTensorFromArray(dec, b.dimension(0), b.dimension(1), b.dimension(2));
	cout << endl << "Decompressed Core: " << endl << decompressedB << endl;

}


void DecodeRleTesting() {
	vector<vector<int>> rle = { {2,4},{0,0,2,1},{0,2},{} };//{ {0,1,1,2},{1,1},{0,1},{0} };//{ {0,6},{1,4},{1,3},{0}};
	vector<vector<bool>> verb = { {},{0},{1,0,0,0},{1,0,0} };//{ {},{1,0,0},{0,1,0,1,1},{0,0,1,0,1,0} };//{ {},{1}, {1,1},{1,0,1} };
	vector<bool> signs;

	double* dec =TTHRESHEncoding::decodeRLE(rle,verb,7,1, signs);
	
}

void readTesting() {

	char txt[] = "AC.txt";
	BitIO::openWrite(txt);
	
	BitIO::writeBit(1, 16);
	BitIO::writeBit(2, 16);
	BitIO::writeBit(3, 16);
	BitIO::writeBit(1, 1);
	BitIO::writeBit(122, 64);

	BitIO::writeRemainingBit();

	BitIO::closeWrite();

	BitIO::openRead(txt);

	int d1 = BitIO::readBit(16);
	int d2 = BitIO::readBit(16);
	int d3 = BitIO::readBit(16);
	int d5 = BitIO::readBit(1);
	uint64_t d4 = BitIO::readBit(64);

	cout << d1 << " " << d2<<" "<<d3<<" "<<d5<<" "<<d4 << endl;

	BitIO::closeRead();

	/*FILE *fp = fopen("erg.txt", "r");

	unsigned short sizes[3];
	fread((void*)sizes, 3, sizeof(unsigned short), fp);

	cout << endl << "Read File Tests: "<<endl<< "D1: " << int(sizes[0]) << " D2: " << int(sizes[1]) << " D3: " << int(sizes[2])<< endl;

	uint64_t scale[1];
	fread((void*)scale, 1, sizeof(uint64_t), fp);

	double ScaleFactor;
	memcpy(&ScaleFactor, (void*)&scale[0], sizeof(scale[0])); //cast back to double

	cout << "ReadScale: " << ScaleFactor<< endl;

	fclose(fp);*/
}

void readTesting2() {

	char txt[] = "erg.txt";
	BitIO::openRead(txt);

	int d1 = BitIO::readBit(sizeof(unsigned short) * 8);
	int d2 = BitIO::readBit(sizeof(unsigned short) * 8);
	int d3 = BitIO::readBit(sizeof(unsigned short) * 8);

	/*int64_t scale= BitIO::readBit(64);

	double ScaleFactor;
	memcpy(&ScaleFactor, (void*)&scale, sizeof(scale));*/

	cout << endl << "D1: " << d1<< " D2: "<<d2 <<" D3: "<<d3 << endl;

	BitIO::closeRead();
}

void ACTesting(VolInputParser dummy) {

	char txt[] = "AC.txt";
	BitIO::openWrite(txt); //Open document to write into

	//testdaten erstellen, in AC speisen
	vector<int> rleTest = { 0,11 };//1,3,3,7,6,6,7,8,9,10,1,3,3,7,11,8,9,10,11,33,45,77};
	vector<int> rleTest2 = { 11};//99,88,66,88,99,1,2,3,4 };
	vector<int> rleTest00 = {  };
	vector<int> rleTest3 = { 11};//20, 21, 20, 21, 21, 20, 21, 20};
	vector<int> rleTest0 = {  };
	vector<int> rleTest4 = { 7,3};//1,2,3,4,5,6,7,8,9,10 };
	vector<int> rleTest5 = { 10};//;99,98,100,1002,1003,2,3,3,4 };
	vector<int> rleTest6 = { 6,3};//1 };

	TTHRESHEncoding::encodeAC(rleTest);
	TTHRESHEncoding::encodeAC(rleTest2);
	TTHRESHEncoding::encodeAC(rleTest00);
	TTHRESHEncoding::encodeAC(rleTest3);
	TTHRESHEncoding::encodeAC(rleTest0);
	TTHRESHEncoding::encodeAC(rleTest4);
	TTHRESHEncoding::encodeAC(rleTest5);
	TTHRESHEncoding::encodeAC(rleTest6);

	BitIO::writeRemainingBit();

	BitIO::closeWrite();

	//decoding
	BitIO::openRead(txt);

	vector<int> decRle;
	TTHRESHEncoding::decodeAC(decRle);

	vector<int> decRle2;
	TTHRESHEncoding::decodeAC(decRle2);

	vector<int> decRle00;
	TTHRESHEncoding::decodeAC(decRle00);

	vector<int> decRle3;
	TTHRESHEncoding::decodeAC(decRle3);

	vector<int> decRle0;
	TTHRESHEncoding::decodeAC(decRle0);

	vector<int> decRle4;
	TTHRESHEncoding::decodeAC(decRle4);

	vector<int> decRle5;
	TTHRESHEncoding::decodeAC(decRle5);

	vector<int> decRle6;
	TTHRESHEncoding::decodeAC(decRle6);

	cout << "Decoding: " << endl;
	for (int i = 0;i < decRle.size();i++) {
		std::cout << decRle[i]<<endl;
	}

	cout << "Decoding2: " << endl;
	for (int i = 0;i < decRle2.size();i++) {
		std::cout << decRle2[i] << endl;
	}

	cout << "Decoding3: " << endl;
	for (int i = 0;i < decRle3.size();i++) {
		std::cout << decRle3[i] << endl;
	}

	cout << "Decoding4: " << endl;
	for (int i = 0;i < decRle4.size();i++) {
		std::cout << decRle4[i] << endl;
	}

	cout << "Decoding5: " << endl;
	for (int i = 0;i < decRle5.size();i++) {
		std::cout << decRle5[i] << endl;
	}

	cout << "Decoding6: " << endl;
	for (int i = 0;i < decRle6.size();i++) {
		std::cout << decRle6[i] << endl;
	}

	BitIO::closeRead();
}

void AcTesting2() {
	char txt[] = "AC.txt";
	BitIO::openWrite(txt); //Open document to write into
	vector<vector<int>> hehe;

	//testdaten erstellen, in AC speisen
	vector<int> rleTest = { 0,11 };//1,3,3,7,6,6,7,8,9,10,1,3,3,7,11,8,9,10,11,33,45,77};
	hehe.push_back(rleTest);
	vector<int> rleTest2 = { 11 };//99,88,66,88,99,1,2,3,4 };
	hehe.push_back(rleTest2);
	vector<int> rleTest00 = {  };
	hehe.push_back(rleTest00);
	vector<int> rleTest3 = { 11 };//20, 21, 20, 21, 21, 20, 21, 20};
	hehe.push_back(rleTest3);
	vector<int> rleTest0 = {  };
	hehe.push_back(rleTest0);
	vector<int> rleTest4 = { 7,3 };//1,2,3,4,5,6,7,8,9,10 };
	hehe.push_back(rleTest4);
	vector<int> rleTest5 = { 10 };//;99,98,100,1002,1003,2,3,3,4 };
	hehe.push_back(rleTest5);
	vector<int> rleTest6 = { 6,3 };//1 };
	hehe.push_back(rleTest6);
	vector<int> rleTest000 = {  };
	hehe.push_back(rleTest000);

	TTHRESHEncoding::encodeACVektor(hehe);

	BitIO::writeBit(32, 64);

	BitIO::writeRemainingBit();

	BitIO::closeWrite();


	BitIO::openRead(txt); //Open document to write into
	vector<vector<int>> decRleB;
	TTHRESHEncoding::decodeACVektor(decRleB);

	int testNum = BitIO::readBit(64);

	BitIO::closeRead();

	cout << decRleB[7][1]<<" " <<decRleB[8].size()<<" "<< testNum << endl;
}
 
void hosvdWrite(VolInputParser dummy) {
	cout << "starting hosvd" << endl;

	vector<MatrixXd> us(3);
	Tensor<myTensorType, 3> b;

	TensorOperations::HOSVD(dummy.DummyTensor, b, us);

	dummy.writeCharacteristicData(b.dimension(0), b.dimension(1), b.dimension(2), us[0].rows(), us[0].cols(), us[1].rows(), us[1].cols(), us[2].rows(), us[2].cols());//save dimension data

	cout << endl << "Core: " << endl << b << endl;
	cout << endl << "U1: " << endl << us[0] << endl;
	cout << endl << "U2: " << endl << us[1] << endl;
	cout << endl << "U3: " << endl << us[2] << endl;

	vector<std::vector<int>> rle;
	vector<std::vector<bool>> raw;
	vector<bool> signs;
	double scale = 0;
	cout << "starting encode: " << endl;

	TTHRESHEncoding::compress(b, us, 0.0003, TTHRESHEncoding::ErrorType::epsilon, rle, raw, scale, signs);

	BitIO::writeRemainingBit();
	BitIO::closeWrite();
}

void hosvdRead(VolInputParser dummy) {
	
	std::vector<std::vector<int>> rle;
	std::vector<std::vector<bool>> raw;
	double scale;
	std::vector<bool> signs;
	
	dummy.readCharacteristicData();
	dummy.readRleData(rle, raw, scale, signs);

	//TODO read data of factor matrices
	int numNorms = BitIO::readBit(64);
	std::cout << "Read Num Norms: " << numNorms << endl;

	//call decompress

	BitIO::closeRead();
}

int main()
{
	//char hehe[] = "stagbeetle277x277x164.dat";//"stagbeetle832x832x494.dat";
	//VolInputParser parse = VolInputParser(hehe);
	//std::cout << "Debug Data Test: " << parse.TensorData(12, 101, 6) << std::endl;

	VolInputParser dummy = VolInputParser();
	//unfoldTesting(dummy);
	//eigenTesting(dummy);

	//hosvdTesting(dummy);
	//TensorOperations::getSlice(dummy.DummyTensor,1,0);
	//DecodeRleTesting();
	//readTesting();
	//readTesting2();
	//ACTesting(dummy);

	hosvdWrite(dummy);
	//hosvdRead(dummy);
	//AcTesting2();
	return 0;
}
