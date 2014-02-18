// Created by Boris Vidolov on 02/16/2014
// Published under Apache 2.0 licence.
#pragma once
#include <string>
#include <sstream>

namespace FastNets
{
	/* Compares two floating point numbers. */
	bool AreSame(double first, double second)
	{
		return (abs(first - second) < 0.000001);
	}

	/* Compares two floating point numbers. */
	bool AreSame(float first, float second)
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

	//TODO: Create ifdefs or other mechanisms to allow for different output functions:
	template <class T>
	inline static T OutputFunction(T input)
	{
		//Hyperbolic tangent:
		return 1 - (2.0 / (1.0 + exp(2*input)));
	}
}