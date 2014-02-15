// Created by Boris Vidolov on 02/14/2014
// Published under Apache 2.0 licence.
#pragma once
#include <Windows.h>

namespace FasteNets
{

/* Represents a single layer in the network */
template<int INPUT, int OUTPUT, class FloatingPoint = double>
class Layer
{
	_CRT_ALIGN(32) FloatingPoint Weights[OUTPUT][INPUT];
	_CRT_ALIGN(32) FloatingPoint ReverseWeights[INPUT][OUTPUT];//Cache for faster calculation
	_CRT_ALIGN(32) FloatingPoint B[OUTPUT];//Input Bias
	_CRT_ALIGN(32) FloatingPoint C[INPUT];//Output Bias

	CRITICAL_SECTION ReverseLock;
	bool  mReverseWeightsDirty;
private:
	Layer(const Layer&){}//No copy

public:

	Layer(void)
	{
	}

	~Layer(void)
	{
	}
};

}//Fast nets
