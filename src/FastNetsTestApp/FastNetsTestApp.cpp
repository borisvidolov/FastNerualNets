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
		cout << "Succeeded" << endl;
		//Networks:
		cout << "Network constructor...";
		Net<8> dummy;
		Net<16, Net<8, Net<8, Net<8>>>> n;
		cout << "Succeeded." << endl;
		cout << "Network writing and reading...";
	}
	catch(string error)
	{
		cout << endl << "Failed: " << error.c_str() << endl;
	}
}

