// Created by Boris Vidolov on 02/15/2014
// Published under Apache 2.0 licence.
#pragma once
#include <vector>
#include <algorithm>

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
			bool operator < (const IndividualStorage& other) const { return mError < other.mError; }
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
				GenerateInitial();
				return;
			}


			throw std::string("Implement me");
		}

		void Select()
		{
			std::stable_sort(mPopulation.begin(), mPopulation.end());
			int maxRemaining = mMaxCount*mSurvivalRate;
			mPopulation.erase(mPopulation.begin() + maxRemaining, mPopulation.end());
		}

		void Train(FloatingPoint* inputMatrix, FloatingPoint* outputMatrix, FloatingPoint* expectedMatrix, int numInputs)
		{
			Populate();
			Evaluate(inputMatrix, outputMatrix, numInputs);
			Select();
		}

		void Evaluate(FloatingPoint* inputMatrix, FloatingPoint* outputMatrix, FloatingPoint* expectedMatrix, int numInputs)
		{
			#pragma omp parallel for
			for (int i = 0; i < (int)mPopulation.size(); ++i)
			{
				//TODO: consider implementing a back calculator that does not use OMP. OMP is the most efficient
				//when it is applied at the top.
				mPopulation[i].mpIndividual->BatchProcessInputFast(inputMatrix, outputMatrix, numInputs);
				mPopulation[i].mError = mPopulation[i].mpIndividual->CalculateErrors(outputMatrix, expectedMatrix, numInputs);
			}
		}

		~Population()
		{
			//Clear the memory:
			for (unsigned i = 0; i < mPopulation.size(); ++i)
			{
				delete mPopulation[i].mpIndividual;
			}
		}
	protected:

		void GenerateInitial()
		{
			//Initial flow, just generate them all:
			mPopulation.reserve(mMaxCount);
			//Add dummy elements:
			for (unsigned i = 0; i < mMaxCount; ++i)
			{
				mPopulation.push_back(IndividualStorage());
			}

			double defaultError = 1e10;
			//Now set these elements in parallel:
			#pragma omp parallel for
			for (int i = 0; i < (int)mPopulation.size(); ++i)
			{
				mPopulation[i].mError = defaultError;
				mPopulation[i].mpIndividual = new Individual();
			}
		}
	};
}