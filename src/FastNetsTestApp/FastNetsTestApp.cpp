// Created by Boris Vidolov on 02/14/2014
// Published under Apache 2.0 licence.
// FastNetsTestApp.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include "..\FastNetsLibrary\Net.h"

using namespace FastNets;
using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	try
	{
		//Tests major scenarios:
		//Layers:
		cout << "Layer Constructor...";
		Layer<8, 8> l1;
		cout << "Succeeded" << endl;

		try
		{
			cout << "Layer Constructor with missing file...";
			Layer<8, 8> l("missing file");
			cout << "Failed";
			return 1;
		}
		catch (std::string error)
		{
			cout << "Succeeded" << endl;
		}

		cout << "Layer Write and read from file...";
		l1.WriteToFile("foo");
		Layer<8, 8> l2("foo");
		remove("foo");
		if (!l1.IsSame(l2))
			throw std::string("Unpersisted layer is different.");
		cout << "Succeeded" << endl;

		//Networks:
		cout << "Network constructor...";
		Net<8> dummy;
		const unsigned input = 16;
		const unsigned output = 8;
		Net<input, Net<8, Net<8, Net<output>>>> n;
		cout << "Succeeded." << endl;

		cout << "Network writing and reading...";
		n.WriteToFile("bar");
		Net<input, Net<8, Net<8, Net<output>>>> n1("bar");
		remove("bar");
		if (!n.IsSame(n1))
			throw std::string("The networks are different.");

		cout << "Succeeded." << endl;

		cout << "Check single slow calculation...";
		_CRT_ALIGN(32) double inputArray[input];
		_CRT_ALIGN(32) double outputArray[output];
		for (unsigned i = 0; i < input; ++i)
		{
			inputArray[i] = i*0.001;
		}
		n.ProcessInputSlow(inputArray, outputArray);
		cout << "Succeeded." << endl;

		cout << "Measure slow calculation...";
		time_t start = time(NULL);
		for (int i = 0; i < 1000000; ++i)
		{
			n.ProcessInputSlow(inputArray, outputArray);
		}
		time_t end = time(NULL);
		cout << "Took: " << (end - start) << " seconds.";

		cout << "Measure AVX calculation...";
		start = time(NULL);
		for (int i = 0; i < 1000000; ++i)
		{
			n.ProcessInputFast(inputArray, outputArray);
		}
		end = time(NULL);
		cout << "Took: " << (end - start) << " seconds.";
	}
	catch(string error)
	{
		cout << endl << "Failed: " << error.c_str() << endl;
	}
}

