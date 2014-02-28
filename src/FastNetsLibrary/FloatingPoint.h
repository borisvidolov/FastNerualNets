// Created by Boris Vidolov on 02/16/2014
// Published under Apache 2.0 licence.
#pragma once
#include <string>
#include <sstream>

#define TANH_OUTPUT

namespace FastNets
{
	/* Compares two floating point numbers. */
	inline bool AreSame(double first, double second)
	{
		return (abs(first - second) < 0.000001);
	}

	/* Compares two floating point numbers. */
	inline bool AreSame(float first, float second)
	{
		return (abs(first - second) < 0.001);
	}

	template<class T>
	bool AreSame(T* data1, T* data2, unsigned count)
	{
		for (unsigned i = 0; i < count; ++i)
		{
			if (!AreSame(data1[i], data2[i]))
				return false;
		}
		return true;
	}

#ifdef TANH_OUTPUT
	//TODO: Create ifdefs or other mechanisms to allow for different output functions:
	template <class T>
	inline static T OutputFunction(T input)
	{
		//Hyperbolic tangent:
		return 1 - (2.0 / (1.0 + exp(2*input)));
	}

	/* calculates the derivative as a function of the regular output */
	template <class T>
	inline static T DerivativeFunction(T output)
	{
		return (1 - output)*(1 + output);
	}

#else
	#error Only thah is implemented for now
#endif

	//Given an integer, returns the closest >= one that is 32 byte aligned.
	inline unsigned AVXAlign(unsigned value)
	{
		return (value + 31) & ~31;
	}

	void TransferAlignedInput(const double* pInput, unsigned inputFeatures, unsigned numInputs, double* avxAlignedOutput);

	/* Calculates the output of a layer. */
	void ProcessInputAVX(double* input, double* output, unsigned inputSize, unsigned outputSize, double* weights, double* bias);
}