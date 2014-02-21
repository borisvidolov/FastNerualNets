// Created by Boris Vidolov on 02/14/2014
// Published under Apache 2.0 licence.
// FastNetsTestApp.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include "..\FastNetsLibrary\Net.h"
#include "..\FastNetsLibrary\Timer.h"

using namespace FastNets;
using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	const unsigned input = 167;
	const unsigned output = 9;
	const unsigned iterations = 100000;	
	unsigned inputSize = (iterations*AVXAlign(input));
	double* inputArray = (double*)_aligned_malloc(inputSize*sizeof(double), 32);
	for (unsigned i = 0; i < inputSize; ++i)
	{
		//Initial initialization to invalid number:
		inputArray[i] = _Nan._Double;
	}
	double* slowOutputArray = (double*)_aligned_malloc((iterations*output)*sizeof(double), 32);
	double* fastOutputArray = (double*)_aligned_malloc((iterations*output)*sizeof(double), 32);
	try
	{
		//Tests major scenarios:
		//Layers:
		cout << "Layer Constructor...";
		Layer<9, 1> l1;
		cout << "Succeeded" << endl;

		try
		{
			cout << "Layer Constructor with missing file...";
			Layer<9, 1> l("missing file");
			cout << "Failed";
			return 1;
		}
		catch (std::string error)
		{
			cout << "Succeeded" << endl;
		}

		cout << "Layer Write and read from file...";
		l1.WriteToFile("foo");
		Layer<9, 1> l2("foo");
		remove("foo");
		if (!l1.IsSame(l2))
			throw std::string("Unpersisted layer is different.");
		cout << "Succeeded" << endl;

		//Networks:
		cout << "Network constructor...";
		Net<8> dummy;
		Net<input, Net<112, Net<112, Net<output>>>> n;
		cout << "Succeeded." << endl;
		
		cout << "Network writing and reading...";
		n.WriteToFile("bar");
		Net<input, Net<112, Net<112, Net<output>>>> n1("bar");
		remove("bar");
		if (!n.IsSame(n1))
			throw std::string("The networks are different.");

		cout << "Succeeded." << endl;
		int alignedInput = AVXAlign(input);
		for (unsigned j = 0; j < iterations; ++j)
		{
			for (unsigned i = 0; i < input; ++i)
			{
				inputArray[j*alignedInput + i] = i*0.0001;
			}
		}

		cout << "Measure slow calculation...";
		{
			Timer t;
			n.BatchProcessInputSlow((double*)inputArray, (double*)slowOutputArray, iterations);
		}

		{
			Timer t;
			cout << "Measure AVX and multi-threaded OMP calculation...";
			n.BatchProcessInputFast((double*)inputArray, (double*)fastOutputArray, iterations);
		}

		cout << "Verifying correctness...";
		if (!AreSame(slowOutputArray, fastOutputArray, output))
			throw std::string("Different results");
		cout << "Succeeded." << endl;

		{
			cout << "Verifying merging of two...";
			Net<5, Net<6, Net<3>>> nFirst, nSecond;
			nFirst.ProcessInputFast(inputArray, fastOutputArray);
			Randomizer<> r;
			Net<5, Net<6, Net<3>>> nSame(nFirst, nFirst, r);
			nSame.ProcessInputFast(inputArray, slowOutputArray);
			if (!AreSame(slowOutputArray, fastOutputArray, output))
				throw std::string("Different results");	

			Net<5, Net<6, Net<3>>> nDifferent(nFirst, nSecond, r);
			nDifferent.ProcessInputFast(inputArray, slowOutputArray);
			if (AreSame(slowOutputArray, fastOutputArray, output))
				throw std::string("Should be different!");	
			cout << "Succeeded." << endl;
		}
	}
	catch(string error)
	{
		cout << endl << "Failed: " << error.c_str() << endl;
	}
	_aligned_free(inputArray);
	_aligned_free(slowOutputArray);
	_aligned_free(fastOutputArray);
}

