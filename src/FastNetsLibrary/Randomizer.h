// Created by Boris Vidolov on 02/19/2014
// Published under Apache 2.0 licence.
#pragma once

#include <Windows.h>
#include <stdio.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

namespace FastNets
{
	/* Helper class for generating random numbers */
	template <class Distribution = boost::random::uniform_int_distribution<>>
	class Randomizer
	{
		unsigned mMax;
		boost::mt11213b mGen;
		Distribution mDist;	
	private:
		Randomizer(const Randomizer&){}//No copy
	public:
		Randomizer(unsigned max = 10000)
			:mMax(max), mDist(1, max)
		{
			mGen.seed((unsigned __int32)time(NULL));
		}
		unsigned Max() const { return mMax; }

		//Returns a random number:
		int Next() { return mDist(mGen); }
		//Returns (0..1)
		double BiasNext() { return ((double)Next())/mMax; }
		//Returns (-absMax, absMax):
		double RangeNext(double absMax){ return absMax*(2*BiasNext() - 1); }
		//Returns (1 - offsetMax, 1 + offsetMax)
		double OffsetNext(double offsetMax){ return 1.0 + RangeNext(offsetMax); }
	};
}