#pragma warning(disable : 4996)
#define _USE_MATH_DEFINES

#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <chrono>

#include "VolInputParser.h"
#include "TensorOperations.h"
#include "TTHRESHEncoding.h"
#include "TensorTruncation.h"
#include <fstream>

#include "HuffmanCode.h"
#include <random>
#include <chrono>

using namespace Eigen;
using namespace std;


void MainWriteTTHRESHTruncation(VolInputParser parser, double targetError) {

	double HybridTime = 0;
	std::chrono::high_resolution_clock::time_point timenow = std::chrono::high_resolution_clock::now(); //time measurements

	char txt[] = "erg.txt";
	BitIO::openWrite(txt);

	cout << "starting hosvd" << endl;

	vector<MatrixXd> us(3);
	Tensor<myTensorType, 3> b;

	TensorOperations::HOSVD(parser.TensorData, b, us);

	TensorTruncation::TruncateTensorTTHRESH(b, us, targetError); //debugging: TE=0.0003

	BitIO::writeRemainingBit();
	BitIO::closeWrite();

	HybridTime += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count() / 1000.;
	cout << "Hybrid time (ms):" << HybridTime << endl;
}

void MainReadTTHRESHTruncation(VolInputParser parser) {

	double HybridReconstructTime = 0;
	std::chrono::high_resolution_clock::time_point timenow = std::chrono::high_resolution_clock::now(); //time measurements

	std::cout << "start reading" << std::endl;
	char txt[] = "erg.txt";
	BitIO::openRead(txt);

	//read in characteristic info
	int d1 = BitIO::readBit(32);
	int d2 = BitIO::readBit(32);
	int d3 = BitIO::readBit(32);
	int r1 = BitIO::readBit(32);
	int r2 = BitIO::readBit(32);
	int r3 = BitIO::readBit(32);

	std::vector<std::vector<int>> rleB;
	std::vector<std::vector<bool>> rawB;
	double scaleB;
	std::vector<bool> signsB;

	std::vector<std::vector<std::vector<int>>> usRle;
	std::vector<std::vector<std::vector<bool>>> usRaw;
	std::vector<double> usScales;
	std::vector<std::vector<bool>> usSigns;

	parser.readRleData(rleB, rawB, scaleB, signsB);
	parser.readNormData();

	for (int i = 0; i < 3; i++) {
		usRle.push_back(std::vector < std::vector<int>>());
		usRaw.push_back(std::vector < std::vector<bool>>());
		usScales.push_back(0);
		usSigns.push_back(std::vector<bool>());

		parser.readRleData(usRle[i], usRaw[i], usScales[i], usSigns[i]);
	}

	//decompress data
	std::cout << "Decompress Data" << std::endl;

	double * decB = TTHRESHEncoding::decodeRLE(rleB, rawB, r1*r2*r3, scaleB, signsB);
	Eigen::Tensor<myTensorType, 3> decompressedB = TensorOperations::createTensorFromArray(decB, r1, r2, r3);

	vector<MatrixXd> us;

	double * decU1 = TTHRESHEncoding::decodeRLE(usRle[0], usRaw[0], r1*d1, usScales[0], usSigns[0]);
	Eigen::MatrixXd decMU1 = TensorOperations::createMatrixFromArray(decU1, d1, r1);
	us.push_back(decMU1);

	double * decU2 = TTHRESHEncoding::decodeRLE(usRle[1], usRaw[1], r2*d2, usScales[1], usSigns[1]);
	Eigen::MatrixXd decMU2 = TensorOperations::createMatrixFromArray(decU2, d2, r2);
	us.push_back(decMU2);

	double * decU3 = TTHRESHEncoding::decodeRLE(usRle[2], usRaw[2], r3*d3, usScales[2], usSigns[2]);
	Eigen::MatrixXd decMU3 = TensorOperations::createMatrixFromArray(decU3, d3, r3);
	us.push_back(decMU3);

	for (int i = 0;i < 3;i++) { //scale back Factor Matrices from Core-Slice Norms
		for (int j = 0;j < us[i].cols();j++) {
			if (parser.tData.coreSliceNorms[i][j] == 0) {
				us[i].col(j).setZero();
			}
			else {
				us[i].col(j) /= parser.tData.coreSliceNorms[i][j]; //TODO norm =0?
			}
		}
	}

	TensorTruncation::RetruncateTensorTTHRESH(decompressedB, us, d1, d2, d3, r1, r2, r3);

	Eigen::Tensor<myTensorType, 3> decompressed;
	TensorOperations::decompress_HOSVD(decompressedB, decompressed, us); //final Tensor is now in decompressed

	BitIO::closeRead();

	HybridReconstructTime += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count() / 1000.;
	cout << "hybrid Reconstruction time (ms):" << HybridReconstructTime << endl;

	//Debugging: Write into raw files
	/*char ErgTxt[] = "result.raw";
	BitIO::openWrite(ErgTxt);
	for (int z = 0;z < decompressed.dimension(2);z++) {
		for (int y = 0; y < decompressed.dimension(0);y++) {
			for (int x = 0;x < decompressed.dimension(1);x++) {
				short dat = short(decompressed(y, x, z));
				BitIO::writeBit(dat, 16);
			}
		}
	}
	BitIO::writeRemainingBit();
	BitIO::closeWrite();*/

}

void MainTruncationWrite(VolInputParser parser, double targetError) {

	double TruncTime = 0;
	std::chrono::high_resolution_clock::time_point timenow = std::chrono::high_resolution_clock::now(); //time measurements

	cout << "starting hosvd" << endl;

	vector<MatrixXd> us(3);
	Tensor<myTensorType, 3> b;

	TensorOperations::HOSVD(parser.TensorData, b, us);

	//write Data for reconstruction
	char txt[] = "erg.txt";
	BitIO::openWrite(txt);

	BitIO::writeBit(uint64_t(b.dimension(0)), 32);
	BitIO::writeBit(uint64_t(b.dimension(1)), 32);
	BitIO::writeBit(uint64_t(b.dimension(2)), 32);


	//calculate Truncation
	std::cout << "starting truncation" << endl;
	TensorTruncation::CalculateTruncation(b, us, targetError);//debugging: TE=0.0003


	BitIO::writeRemainingBit();
	BitIO::closeWrite();

	TruncTime += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count() / 1000.;
	cout << "Truncation time (ms):" << TruncTime << endl;

}

void MainTruncationRead(VolInputParser parser) {

	double TruncationReconstructTime = 0;
	std::chrono::high_resolution_clock::time_point timenow = std::chrono::high_resolution_clock::now(); //time measurements

	std::cout << "start reading" << std::endl;
	char txt[] = "erg.txt";
	BitIO::openRead(txt);

	int d1 = BitIO::readBit(32);
	int d2 = BitIO::readBit(32);
	int d3 = BitIO::readBit(32);
	int r1 = BitIO::readBit(32);
	int r2 = BitIO::readBit(32);
	int r3 = BitIO::readBit(32);

	vector<MatrixXd> us(3);
	Tensor<myTensorType, 3> b;

	TensorTruncation::CalculateRetruncation(b, us, d1, d2, d3, r1, r2, r3);

	Eigen::Tensor<myTensorType, 3> decompressed;
	TensorOperations::decompress_HOSVD(b, decompressed, us); //final Tensor is now in decompressed
	//cout << endl << "Decompress: " << endl << decompressed << endl << endl;

	BitIO::closeRead();

	TruncationReconstructTime += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count() / 1000.;
	cout << "Truncation Reconstruction time (ms):" << TruncationReconstructTime << endl;
}


void MainhosvdWrite(VolInputParser parser, double targetError) {
	cout << "starting hosvd" << endl;

	double TTHRESHTime = 0;
	std::chrono::high_resolution_clock::time_point timenow = std::chrono::high_resolution_clock::now(); //time measurements

	vector<MatrixXd> us(3);
	Tensor<myTensorType, 3> b;

	TensorOperations::HOSVD(parser.TensorData, b, us);//calc Tucker decomposition

	parser.writeCharacteristicData(b.dimension(0), b.dimension(1), b.dimension(2), us[0].rows(), us[0].cols(), us[1].rows(), us[1].cols(), us[2].rows(), us[2].cols());//save dimension data

	vector<std::vector<int>> rle;
	vector<std::vector<bool>> raw;
	vector<bool> signs;
	double scale = 0;
	cout << "starting encode: " << endl;

	TTHRESHEncoding::compress(b, us, targetError, TTHRESHEncoding::ErrorType::epsilon, rle, raw, scale, signs, nullptr);//compress Tucker with RLE+AC and write, //debugging: TE=0.0003

	BitIO::writeRemainingBit();
	BitIO::closeWrite();

	TTHRESHTime += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count() / 1000.;
	cout << "TTHRESH time (ms):" << TTHRESHTime << endl;

}

void MainhosvdRead(VolInputParser dummy) {

	double TTHRESHReconstructTime = 0;
	std::chrono::high_resolution_clock::time_point timenow = std::chrono::high_resolution_clock::now(); //time measurements

	std::vector<std::vector<int>> rleB;
	std::vector<std::vector<bool>> rawB;
	double scaleB;
	std::vector<bool> signsB;

	std::vector<std::vector<std::vector<int>>> usRle;
	std::vector<std::vector<std::vector<bool>>> usRaw;
	std::vector<double> usScales;
	std::vector<std::vector<bool>> usSigns;

	//read in data
	std::cout << "Reading Data" << std::endl;

	dummy.readCharacteristicData();
	dummy.readRleData(rleB, rawB, scaleB, signsB);
	dummy.readNormData();

	for (int i = 0; i < 3; i++) {
		usRle.push_back(std::vector < std::vector<int>>());
		usRaw.push_back(std::vector < std::vector<bool>>());
		usScales.push_back(0);
		usSigns.push_back(std::vector<bool>());

		dummy.readRleData(usRle[i], usRaw[i], usScales[i], usSigns[i]);
	}

	//decompress data
	std::cout << "Decompress Data" << std::endl;

	double * decB = TTHRESHEncoding::decodeRLE(rleB, rawB, dummy.tData.dim1*dummy.tData.dim2*dummy.tData.dim3, scaleB, signsB);
	Eigen::Tensor<myTensorType, 3> decompressedB = TensorOperations::createTensorFromArray(decB, dummy.tData.dim1, dummy.tData.dim2, dummy.tData.dim3);

	vector<MatrixXd> us;

	double * decU1 = TTHRESHEncoding::decodeRLE(usRle[0], usRaw[0], dummy.tData.U1R*dummy.tData.U1C, usScales[0], usSigns[0]);
	Eigen::MatrixXd decMU1 = TensorOperations::createMatrixFromArray(decU1, dummy.tData.U1R, dummy.tData.U1C);
	us.push_back(decMU1);

	double * decU2 = TTHRESHEncoding::decodeRLE(usRle[1], usRaw[1], dummy.tData.U2R*dummy.tData.U2C, usScales[1], usSigns[1]);
	Eigen::MatrixXd decMU2 = TensorOperations::createMatrixFromArray(decU2, dummy.tData.U2R, dummy.tData.U2C);
	us.push_back(decMU2);

	double * decU3 = TTHRESHEncoding::decodeRLE(usRle[2], usRaw[2], dummy.tData.U3R*dummy.tData.U3C, usScales[2], usSigns[2]);
	Eigen::MatrixXd decMU3 = TensorOperations::createMatrixFromArray(decU3, dummy.tData.U3R, dummy.tData.U3C);
	us.push_back(decMU3);

	for (int i = 0;i < 3;i++) { //scale back Factor Matrices from Core-Slice Norms
		for (int j = 0;j < us[i].cols();j++) {
			if (dummy.tData.coreSliceNorms[i][j] == 0) {
				us[i].col(j).setZero();
			}
			else {
				us[i].col(j) /= dummy.tData.coreSliceNorms[i][j]; //TODO norm =0?
			}
		}
	}

	//turn back into tensor
	Eigen::Tensor<myTensorType, 3> decompressed;
	TensorOperations::decompress_HOSVD(decompressedB, decompressed, us); //final Tensor is now in decompressed
	//cout << endl << "Decompress: " << endl << decompressed << endl << endl;

	//std::cout << "Debug Data Test: " << decompressed(12, 101, 6) << " " << decompressed(13, 101, 6) << " " << decompressed(14, 101, 6) << " " << decompressed(15, 101, 6)
	//	<< " " << decompressed(16, 101, 6) << " " << decompressed(17, 101, 6) << std::endl;

	std::cout << "Finished Decompression " << std::endl;
	BitIO::closeRead();

	TTHRESHReconstructTime += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count() / 1000.;
	cout << "TTHRESH Reconstruction time (ms):" << TTHRESHReconstructTime << endl;

	//DEBUGGING: write erg into raw file
	/*char ErgTxt[] = "result.raw";
	BitIO::openWrite(ErgTxt);
	for (int z = 0;z < decompressed.dimension(2);z++) {
		for (int y = 0; y < decompressed.dimension(0);y++) {
			for (int x = 0;x < decompressed.dimension(1);x++) {
				short dat = short(decompressed(y, x, z));
				BitIO::writeBit(dat, 16);
			}
		}
	}
	BitIO::writeRemainingBit();
	BitIO::closeWrite();*/

}


int main(int argc, char*argv[]) {

	if (argc>3) {
		std::cout << "Too many arguments!" << endl;
		return 1;
	}

	VolInputParser parse = VolInputParser(argv[1]);
	double targetError = atof(argv[2]);

	int input = 0;
	std::cout << "Type	 1 for original TTHRESH,\n	 2 for core truncation,\n	 3 for TTHRESH-Truncation Hybrid" << std::endl;
	std::cin >> input;

	switch (input)
	{
	case 1: cout << "TTHRESH init" << endl;
			//std::cout << "Debug Data Test: " << parse.TensorData(12, 101, 6) << " " << parse.TensorData(13, 101, 6) << " " << parse.TensorData(14, 101, 6) << " " << parse.TensorData(15, 101, 6)
			//<< " " << parse.TensorData(16, 101, 6) << " " << parse.TensorData(17, 101, 6) << std::endl;
			MainhosvdWrite(parse, targetError);
			MainhosvdRead(parse);
		break;

	case 2: cout << "Core Trunc init" << endl;
		MainTruncationWrite(parse, targetError);
		MainTruncationRead(parse);
		break;

	case 3: cout << "Hybrid init" << endl;
		MainWriteTTHRESHTruncation(parse, targetError);
		MainReadTTHRESHTruncation(parse);
		break;

	default: cout << "False input" << endl;
		return 1;
		break;
	}
	
}