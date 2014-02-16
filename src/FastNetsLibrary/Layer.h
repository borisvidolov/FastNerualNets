// Created by Boris Vidolov on 02/14/2014
// Published under Apache 2.0 licence.
#pragma once
#include <Windows.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <omp.h>
#include <time.h>

namespace FasteNets
{

/* Represents a single layer in the network. Note that the class will
initialize the OMP threads to achieve maximum performance gain.*/
template<int INPUT, int OUTPUT, class FloatingPoint = double>
class Layer
{
	_CRT_ALIGN(32) FloatingPoint mWeights[OUTPUT][INPUT];
	_CRT_ALIGN(32) FloatingPoint mReverseWeights[INPUT][OUTPUT];//Cache for faster calculation
	_CRT_ALIGN(32) FloatingPoint mB[OUTPUT];//Input Bias
	_CRT_ALIGN(32) FloatingPoint mC[INPUT];//Output Bias

	bool  mReverseWeightsDirty;
private:
	Layer(const Layer&){}//No copy

public:

	Layer(void)
	{
		ValidateTemplateParameters();
		InitializeOmp();

		boost::mt11213b gen;
		gen.seed((unsigned __int32)time(NULL));
		const int max = 10000;
		boost::random::uniform_int_distribution<> dist(1, max);
		for (int i = 0; i < OUTPUT; ++i)
		{
			for (int j = 0; j < INPUT; ++j)
			{ 
				mWeights[i][j] = GetRandomWeight(dist, gen, max);
			}
		}
		for (int i = 0; i < OUTPUT; ++i)
			mB[i] = GetRandomWeight(dist, gen, max);
		for (int i = 0; i < INPUT; ++i)
			mC[i] = GetRandomWeight(dist, gen, max);
		mReverseWeightsDirty = true;	
	}

	~Layer(void)
	{
	}

protected:
	//Compile-time checks on the parameters
	void ValidateTemplateParameters();

	double GetRandomWeight(boost::random::uniform_int_distribution<>& dist, boost::mt11213b& gen, const int max )
	{
		double value = 12*((double)dist(gen)/max) - 6;
		if (abs(value) < 0.001)//Don't create links with 0 strength
			value = 0.1;	
		return value;// /Input;
	}

	void InitializeOmp()
	{
		size_t bufferSize;
		char* pBuffer = NULL;
		errno_t res = _dupenv_s(&pBuffer, &bufferSize, "NUMBER_OF_PROCESSORS1");
		if (res || !bufferSize || !pBuffer)
		{
			std::cout << std::endl << "Cannot determine the number of logical processors. Error: " << res;
			return;
		}
		pBuffer[bufferSize - 1] = 0;//Just in case
		int numCPUs = atoi(pBuffer);
		if (numCPUs != LONG_MIN)
		{
			omp_set_num_threads(numCPUs);
		}
		free(pBuffer);
	}
};

#pragma warning (push)
#pragma warning (disable:4101)
template<int INPUT, int OUTPUT, class FloatingPoint>
void Layer<INPUT, OUTPUT, FloatingPoint>::ValidateTemplateParameters()
{
	//Compile-time checks to ensure that the parameters meet the expectations
	int x[!(INPUT % 8)];//Ensure division by 8
	int y[!(OUTPUT % 8)];//Ensure division by 8
}
#pragma warning (pop)

}//Fast nets

