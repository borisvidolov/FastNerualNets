// Created by Boris Vidolov on 02/14/2014
// Published under Apache 2.0 licence.
// FastNetsTestApp.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include "..\FastNetsLibrary\Net.h"
#include "..\FastNetsLibrary\Timer.h"
#include "..\FastNetsLibrary\Genetic.h"

using namespace FastNets;
using namespace std;

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
		Layer<9, 1> l1(true);
		cout << "Succeeded" << endl;

		try
		{
			cout << "Layer Constructor with missing file...";
			Layer<9, 1> l(false);
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
		Layer<9, 1> l2(false);
		l2.ReadFromFile("foo");
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
			Net<5, Net<6, Net<3>>> nFirst, nSecond;
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

		{
			typedef Net<2, Net<2, Net<1>>> NetType;
			cout << "Test genetic algos...";
			Population<NetType> population(10000, 0.01);
			double xorInput[] ={-1, -1,
								-1, 1,
								 1, -1,
								 1, 1};
			double xorExpected[] = { -0.5, 0.5, 0.5, -0.5 };
			AlignedMatrix<2> xorInputMatrix(xorInput, 4);
			AlignedMatrix<1> xorExpectedMatrix(xorExpected, 4);
			cout << endl;
			double previousError = 1e10;
			{
				Timer t;
				for (int i = 0; i < 100; ++i)
				{
					double error = population.Train(xorInputMatrix, xorExpectedMatrix, 0.1, true);
					cout << "Iteration: " << i << "; Error: " << error << endl;
					if (error > previousError)
						throw std::string("Not improving");
					previousError = error;
				}
			}

			cout << "Succeeded." << endl;
		}
	}
	catch(string error)
	{
		cout << endl << "Failed: " << error.c_str() << endl;
	}
}

