// Created by Boris Vidolov on 02/14/2014
// Published under Apache 2.0 licence.
#pragma once

#include <Windows.h>
#include <stdio.h>

namespace FastNets
{
/* A simple to use timer class that prints out the duration in the output window.
	Example:
	{
		Timer t;
		... //Do work
	}//The timer destructor will print here something like "Took 5.35s\n"*/
class Timer
{
protected:
	ULONGLONG mStart;
public:
	Timer(){ mStart = GetTickCount64(); }
	~Timer()
	{ 
		ULONGLONG result = GetTickCount64() - mStart;
		double seconds = result/1000.0;
		printf("\nTook %.2fs\n", seconds);
	}
};
}//Namespace FastNets