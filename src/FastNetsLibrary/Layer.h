// Created by Boris Vidolov on 02/14/2014
// Published under Apache 2.0 licence.
#pragma once
#include <Windows.h>
#include <boost/random/mersenne_twister.hpp>
#include <omp.h>

namespace FasteNets
{

/* Represents a single layer in the network. Note that the class will
initialize the OMP threads to achieve maximum performance gain.*/
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
		ValidateTemplateParameters();
		InitializeOmp();
		InitializeCriticalSection(&global);
		boost::mt11213b gen;
		gen.seed((unsigned __int32)time(NULL));
		const int max = 10000;
		boost::random::uniform_int_distribution<> dist(1, max);
		for (int i = 0; i < Output; ++i)
			for (int j = 0; j < Input; ++j)
			{ 
				Weights[i][j] = GetRandomWeight(dist, gen, max);
			}
		for (int i = 0; i < Output; ++i)
			B[i] = GetRandomWeight(dist, gen, max);
		for (int i = 0; i < Input; ++i)
			C[i] = GetRandomWeight(dist, gen, max);
		mReverseWeightsDirty = true;	
	}

	~Layer(void)
	{
	}

protected:
	//Compile-time checks on the parameters
	void ValidateTemplateParameters();

	void InitializeOmp()
	{
		char buffer[10];
		int bufferSize = sizeof(buffer);
		errno_t res = _dupenv_s(&buffer, &bufferSize, "NUMBER_OF_PROCESSORS1");
		if (res)
		{
			std::cout << std::endl << "Cannot determine the number of logical processors. Error: " << res;
			return;
		}
		buffer[sizeof(buffer)/sizeof(buffer[0]) - 1] = 0;//Just in case
		int numCPUs = atoi(buffer);
		if (numCPUs != LONG_MIN)
		{
			omp_set_num_threads(numCPUs);
		}
	}
};

#pragma warning (push)
#pragma warning (disable:4101)
template<int INPUT, int OUTPUT, class FloatingPoint>
void Layer<INPUT, OUTPUT, FloatingPoint>::ValidateTemplateParameters()
{
	//Compile-time checks to ensure that the parameters meet the expectations
	int x[!(Input % 8)];//Ensure division by 8
	int y[!(Output % 8)];//Ensure division by 8
}
#pragma (pop)

}//Fast nets
