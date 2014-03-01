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
	bool AreSame(const T* data1, const T* data2, unsigned count)
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
#define AVXAlignBytes(X) ((X + 31) & ~31);
//Given an integer number of elements (e.g. doubles) and their bytes size
//returns the 32 byte aligned number. E.g. "AVXAlign(3, sizeof(double))" equates to 4.
#define AVXAlignType(X, size) ((X + (32/size) - 1) & ~(32/size - 1));

	template<class T>
	unsigned AVXAlign(unsigned numElements)
	{
		return AVXAlignType(numElements, sizeof(T));
	}

	//Transferred a matrix, containing non-aligned inputs into aligned one
	template <class FloatingPointType>
	void TransferAlignedInput(const FloatingPointType* pInput, unsigned inputFeatures, unsigned numInputs, FloatingPointType* pAvxAlignedOutput)
	{
		const unsigned alignedRow = AVXAlign<FloatingPointType>(inputFeatures);
		for (unsigned i = 0; i < numInputs; ++i)
		{
			memcpy(pAvxAlignedOutput + i*alignedRow, pInput + i*inputFeatures, inputFeatures*sizeof(FloatingPointType));
		}
	}

	/* Calculates the output of a layer. */
	void ProcessInputAVX(const double* input, double* output, unsigned inputSize, unsigned outputSize, const double* weights, const double* bias);
}