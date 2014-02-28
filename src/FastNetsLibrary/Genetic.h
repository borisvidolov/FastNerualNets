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

		//Returns whether this is the initial population
		bool Populate()
		{
			unsigned populationCount = mPopulation.size();
			if (!populationCount)
			{
				GenerateInitial();
				return true;
			}

			//A simple algorithm to create a population, where the most
			//successful parents have the most children. Success is measured by how
			//small is the error (how much different it is from the maximum error):
			double maxError = 1e100;
			for (unsigned i = 0; i < populationCount; ++i)
			{
				double error = mPopulation[i].mError; 
				if (maxError < error)
					maxError = error;
			}

			double totalError = 0;
			for (unsigned i = 0; i < populationCount; ++i)
			{
				totalError += mPopulation[i].mError;
			}
			unsigned populationToAdd = mMaxCount - populationCount;
			mPopulation.insert(mPopulation.end(), populationToAdd, IndividualStorage());

			double totalSuccess = populationCount*maxError - totalError;//Reverse it

			double distributedError = (populationCount - 1)*totalError;
			unsigned currentPlace = populationCount;
			Randomizer<> rand;
			for (unsigned int i = 0; i < populationCount - 1; ++i)
			{
				for (unsigned int j = i + 1; j < populationCount; ++j)
				{
					const IndividualStorage& first = mPopulation[i];
					const IndividualStorage& second = mPopulation[j];
					double combinedSuccess = 2*maxError - (first.mError + second.mError);
					int numChildren = (int)((combinedSuccess/totalSuccess)*populationToAdd);
					for (int k = 0; k < numChildren; ++k)
					{
						mPopulation[populationCount++].mpIndividual = new Individual(*first.mpIndividual, *second.mpIndividual, rand);
					}
				}
			}

			return false;
		}

		void Select()
		{
			std::stable_sort(mPopulation.begin(), mPopulation.end());
			int maxRemaining = mMaxCount*mSurvivalRate;
			mPopulation.erase(mPopulation.begin() + maxRemaining, mPopulation.end());
		}

		//Static input parameter means that the Train method will be called always with
		//the same input
		void Train(FloatingPoint* inputMatrix, FloatingPoint* outputMatrix, FloatingPoint* expectedMatrix, int numInputs, bool staticInput)
		{
			bool initial = Populate();
			//The first time we evaluate all elements. Beyond that we only evaluate the new ones, if the input is static:
			Evaluate(inputMatrix, outputMatrix, numInputs, (initial || !staticInput) ? 0 : (int)(mMaxCount*mSurvivalRate));
			Select();
		}

		void Evaluate(FloatingPoint* inputMatrix, FloatingPoint* outputMatrix, FloatingPoint* expectedMatrix, int numInputs, int skipElements = 0)
		{
			#pragma omp parallel for
			for (int i = skipElements; i < (int)mPopulation.size(); ++i)
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
			mPopulation.clear();//Just in case
			//Initial flow, just generate them all:

			//Add dummy elements:
			mPopulation.assign(mMaxCount, IndividualStorage());

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