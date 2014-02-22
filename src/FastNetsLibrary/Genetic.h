// Created by Boris Vidolov on 02/15/2014
// Published under Apache 2.0 licence.
#pragma once
#include <vector>

namespace FastNets
{
	/* Implements the genetic algorithm learning. Example usage:
	Population<Net<2, Net<1>>> n;
	do 
	{
		error = n.Train(xorDataInput, xorDataOutput, 4);
	}
	while (error > 0.3);*/
	template<class Individual, class FloatingPoint = double>
	class Population
	{
	protected:
		struct IndividualStorage
		{
			IndividualStorage():mError(1e10), mpIndividual(NULL){}
			FloatingPoint	mError;
			Individual*		mpIndividual;
		};
		unsigned		mMaxCount;
		double			mSurvivalRate;
		std::vector<IndividualStorage> mPopulation;
	private:
		Population(const Population& other){}//No copy
	public:
		Population(unsigned maxCount, double survivalRate)
			:mMaxCount(maxCount), mSurvivalRate(survivalRate)
		{}

		void Populate()
		{
			if (!mPopulation.size())
			{
				//Initial flow, just generate them all:
				mPopulation.reserve(mMaxCount);
				//Add dummy elements:
				for (unsigned i = 0; i < mMaxCount; ++i)
				{
					mPopulation.push_back(IndividualStorage());
				}
				#pragma omp parallel for
				for (unsigned i = 0; i < mMaxCount; ++i)
				{
					mPopulation.
				}
			}
			throw std::string("Implement me");
		}

		void Select()
		{
			throw std::string("Implement me");
		}

		void Train(FloatingPoint* inputMatrix, FloatingPoint* outputMatrix, int numInputs)
		{
			throw std::string("Implement me");
		}

		~Population()
		{
			throw std::string("Implement me");
		}
	};
}