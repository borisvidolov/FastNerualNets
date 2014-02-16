// Created by Boris Vidolov on 02/14/2014
// Published under Apache 2.0 licence.
// FastNetsTestApp.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include "..\FastNetsLibrary\Layer.h"

using namespace FasteNets;
using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	//Tests major scenarios:
	cout << "Constructor...";
	Layer<8, 8> l1;
	cout << "Succeeded" << endl;
	try
	{
		cout << "Constructor with missing file...";
		Layer<8, 8> l("missing file");
		cout << "Failed";
		return 1;
	}
	catch (std::string error)
	{
		cout << "Succeeded" << endl;
	}
	cout << "Write and read from file...";
	l1.WriteToFile("foo");
	Layer<8, 8> l2("foo");
	remove("foo");
	cout << "Succeeded" << endl;
}

