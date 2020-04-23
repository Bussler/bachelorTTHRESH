#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <fstream>
#include "TensorOperations.h"

class VolInputParser
{
public:
	VolInputParser();
	VolInputParser(char * txtname);
	~VolInputParser();

	//Eigen::MatrixXd MatrixData; //Matrix to store the data
	Eigen::Tensor<myTensorType, 3> TensorData; //Tensor to strore the data

	Eigen::Tensor<myTensorType, 3> DummyTensor;//dummy tensor for testing
	
	void readInputVol(char * txtname);

};

