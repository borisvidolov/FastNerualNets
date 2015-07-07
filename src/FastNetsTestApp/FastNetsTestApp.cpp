// Created by Boris Vidolov on 02/14/2014
// Published under Apache 2.0 licence.
// FastNetsTestApp.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#define FIXED_RANDOM
#include "..\FastNetsLibrary\Net.h"
#include "..\FastNetsLibrary\Timer.h"
#include "..\FastNetsLibrary\Genetic.h"

using namespace FastNets;
using namespace std;

#define TEST_PERF
#define TEST_GENETIC

int _tmain(int argc, _TCHAR* argv[])
{
	const unsigned input = 167;
	const unsigned output = 9;
#ifdef NDEBUG
	const unsigned iterations = 100000;	
#else
	const unsigned iterations = 10000;	
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif
	AlignedMatrix<input> inputMatrix(iterations);
	AlignedMatrix<output> slowOutputMatrix(iterations);
	AlignedMatrix<output> fastOutputMatrix(iterations);

	try
	{
		//Tests major scenarios:
		//Layers:
		cout << "Layer Constructor...";
		Layer<9, 1> l1(InitializeForGenetic);
		cout << "Succeeded" << endl;

		try
		{
			cout << "Layer Constructor with missing file...";
			Layer<9, 1> l(NoWeightsInitialize);
			l.ReadFromFile("missing file");
			cout << "Failed";
			return 1;
		}
		catch (std::string error)
		{
			cout << "Succeeded" << endl;
		}

		cout << "Layer Write and read from file...";
		l1.WriteToFile("foo");
		Layer<9, 1> l2(NoWeightsInitialize);
		l2.ReadFromFile("foo");
		remove("foo");
		if (!l1.IsSame(l2))
			throw std::string("Unpersisted layer is different.");
		cout << "Succeeded" << endl;

		//Networks:
		cout << "Network constructor...";
		Net<8> dummy(InitializeForGenetic);
		Net<input, Net<112, Net<112, Net<output>>>> n(InitializeForGenetic);
		cout << "Succeeded." << endl;
		
		cout << "Network writing and reading...";
		n.WriteToFile("bar");
		Net<input, Net<112, Net<112, Net<output>>>> n1("bar");
		remove("bar");
		if (!n.IsSame(n1))
			throw std::string("The networks are different.");

		cout << "Succeeded." << endl;

#ifdef TEST_PERF
		int alignedInput = AVXAlign<double>(input);
		for (unsigned j = 0; j < iterations; ++j)
		{
			double* pInput = inputMatrix.GetRow(j);
			for (unsigned i = 0; i < input; ++i)
			{
				pInput[i] = i*0.0001;
			}
		}

		cout << "Measure slow calculation...";
		{
			Timer t;
			n.BatchProcessInputSlow(inputMatrix, slowOutputMatrix);
		}

		{
			cout << "Measure AVX and multi-threaded OMP calculation...";
			Timer t;
			n.BatchProcessInputFast(inputMatrix, fastOutputMatrix);
			cout << "Succeeded." << endl;
		}

		{
			cout << "Verify errors...";
			Timer t;
			double error = n.CalculateError(slowOutputMatrix, fastOutputMatrix);
			if (!AreSame(error, 0))
				throw std::string("Wrong error calculation.");
			cout << "Succeeded." << endl;
		}

		cout << "Verifying correctness...";
		if (!slowOutputMatrix.IsSame(fastOutputMatrix))
			throw std::string("Different results");
		cout << "Succeeded." << endl;
#endif

		{
			cout << "Test randomizer...";
			Randomizer<> r1, r2;
			if (r1.Next() == r2.Next())
				throw std::string("Should be different!");
			if (r1.RangeNext(6) == r2.RangeNext(6))
				throw std::string("Should be different!");
			cout << "Succeeded." << endl;
		}

		{
			cout << "Verifying merging of two nets...";
			Net<5, Net<6, Net<3>>> nFirst(InitializeForGenetic), nSecond(InitializeForGenetic);
			nFirst.ProcessInputFast(inputMatrix.GetRow(0), fastOutputMatrix.GetRow(0));
			nSecond.ProcessInputFast(inputMatrix.GetRow(0), slowOutputMatrix.GetRow(0));
			if (AreSame(slowOutputMatrix.GetRow(0), fastOutputMatrix.GetRow(0), nFirst.Output))
				throw std::string("Should be different!");	
			Randomizer<> r;
			Net<5, Net<6, Net<3>>> nSame(nFirst, nFirst, r);
			nSame.ProcessInputFast(inputMatrix.GetRow(0), slowOutputMatrix.GetRow(0));
			if (!AreSame(slowOutputMatrix.GetRow(0), fastOutputMatrix.GetRow(0), nFirst.Output))
				throw std::string("Different results");	

			Net<5, Net<6, Net<3>>> nDifferent(nFirst, nSecond, r);
			nDifferent.ProcessInputFast(inputMatrix.GetRow(0), slowOutputMatrix.GetRow(0));
			if (AreSame(slowOutputMatrix.GetRow(0), fastOutputMatrix.GetRow(0), nFirst.Output))
				throw std::string("Should be different!");	
			cout << "Succeeded." << endl;

			cout << "Verify mutation...";
			nDifferent.Mutate(0.1);
			nDifferent.ProcessInputFast(inputMatrix.GetRow(0), fastOutputMatrix.GetRow(0));
			if (AreSame(slowOutputMatrix.GetRow(0), fastOutputMatrix.GetRow(0), nFirst.Output))
				throw std::string("Should be different!");	
			cout << "Succeeded." << endl;
		}
		cout << "Press enter to continue";
		_gettchar();
	
		typedef Net<2, Net<2, Net<1>>> XorNetType;
#ifdef TANH_OUTPUT
		double testInput[] ={-1, -1,
							 -1, 1,
							  1, -1,
							  1, 1};
		double testExpected[] = { -0.5, 0.5, 0.5, -0.5 };
#elif defined(SIGMOID_OUTPUT)
		double testInput[] ={0, 0,
							 0, 1,
							 1, 0,
							 1, 1};
		double testExpected[] = { 0, 1, 1, 0 };
#else
	#error Fix it yourself :)
#endif
		AlignedMatrix<2> xorInputMatrix(testInput, _countof(testExpected));
		AlignedMatrix<1> xorExpectedMatrix(testExpected, _countof(testExpected));
#ifdef TEST_GENETIC
		{
			cout << "Test genetic algos...";
			Population<XorNetType> population(10000, 0.01);
			cout << endl;
			double previousError = 1e10;
			{
				Timer t;
#  ifdef NDEBUG
				const unsigned generations = 100;
#  else
				const unsigned generations = 100;
#  endif
				for (unsigned i = 0; i < generations && previousError > 1e-4; ++i)
				{
					double error = population.Train(xorInputMatrix, xorExpectedMatrix, 0.3, true);
					cout << "Iteration: " << i << "; Error: " << error << endl;
					if (error > previousError)
						throw std::string("Not improving");
					previousError = error;
				}
			}

			cout << "Succeeded." << endl;
		}
		cout << "Press enter to continue";
		_gettchar();
#endif
		{
			XorNetType net(InitializeForBackProp);
			cout << "Test back propagataion...";
			cout << endl;
			net.PrintWeights();
			double previousError = 1e10;
			{
				Timer t;
				int i = 0;
				AlignedMatrix<1> xorOutputMatrix(_countof(testExpected));
				while(t.Seconds() < 300 && previousError > 1e-4)
				{
					double error = net.BackPropagation(xorInputMatrix, xorExpectedMatrix, 0.3);
					//net.PrintWeights();
					//_gettchar();
					if (!((++i) % 1000))
					{
						cout << "Iteration: " << i << "; Error: " << error << endl;
						//net.PrintWeights();
					}
					/*if (error > previousError)
						throw std::string("Not improving");*/
					previousError = error;
				}
				cout << "Iteration: " << i << "; Error: " << previousError << endl;
			}

			cout << "Succeeded." << endl;
		}
	}
	catch(string error)
	{
		cout << endl << "Failed: " << error.c_str() << endl;
	}
}

