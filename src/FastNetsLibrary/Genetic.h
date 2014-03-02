// Created by Boris Vidolov on 02/15/2014
// Published under Apache 2.0 licence.
#pragma once
#include <vector>
#include <algorithm>
#include <map>
#include <omp.h>

namespace FastNets
{
	/* Implements the genetic algorithm learning. This algorithm handles negative error values
	and works well, even if the error function is not continuous. The algorithm can be used easily
	to cases where we measure discrete success, buy just adding a "-" sign in front of the success
	function. E.g. if you are training the robot to score soccer goals, the success function would
	be the number of goals (nGoals). The error is -nGoals. Another example is trading of stocks where
	the success may be the money made and the same principle can be applied.
	Example usage:
	Population<Net<2, Net<2, Net<1>>>> n;
	do 
	{
		error = n.Train(xorDataInput, xorDataOutput, 0.1, true);
	}
	while (error > 0.01);
	*/
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
		IndividualStorage* mpPopulation;
		bool		    mSelected;//Wheter a first selection has happened
	private:
		Population(const Population& other){}//No copy
	public:
		Population(unsigned maxCount, double survivalRate)
			:mMaxCount(maxCount), mSurvivalRate(survivalRate), mSelected(false)
		{
			mpPopulation = new IndividualStorage[mMaxCount];
			for (unsigned i = 0; i < mMaxCount; ++i)
			{
				mpPopulation[i].mpIndividual = new Individual();
			}
		}

		~Population()
		{
			for (unsigned i = 0; i < mMaxCount; ++i)
			{
				delete mpPopulation[i].mpIndividual;
			}
			delete [] mpPopulation;
		}

		//Returns whether this is the initial population
		bool Populate(double mutationRate)
		{
			if (!mSelected)
			{
				return true;
			}

			//A simple algorithm to create a population, where the most
			//successful parents have the most children. Success is measured by how
			//small is the error (how much different it is from the maximum error):
			unsigned selectionCount = SelectCount();
			double maxError = -1e100;
			for (unsigned i = 0; i < selectionCount; ++i)
			{
				double error = mpPopulation[i].mError; 
				if (maxError < error)
					maxError = error;
			}

			double totalError = 0;
			for (unsigned i = 0; i < selectionCount; ++i)
			{
				totalError += mpPopulation[i].mError;
			}
			unsigned populationToSet = mMaxCount - selectionCount;

			double totalSuccess = selectionCount*maxError - totalError;//Reverse error into success
			double totalPairsSuccess = (selectionCount - 1)*totalSuccess;

			unsigned currentPlace = selectionCount;
			Randomizer<> rand;
			double reminder = 0;
			for (unsigned int i = 0; i < selectionCount - 1; ++i)
			{
				for (unsigned int j = i + 1; j < selectionCount; ++j)
				{
					const IndividualStorage& first = mpPopulation[i];
					const IndividualStorage& second = mpPopulation[j]; 
					double combinedSuccess = 2*maxError - first.mError - second.mError;
					double dNumChildren = ((combinedSuccess/totalPairsSuccess)*populationToSet);
					int numChildren = (int)dNumChildren;
					reminder += dNumChildren - numChildren;
					if (reminder >= 1)
					{
						numChildren++;
						reminder -= 1;
					}
					for (int k = 0; k < numChildren; ++k)
					{
						Individual& rToChange = *mpPopulation[currentPlace++].mpIndividual;
						rToChange.SetFromMergedParents(*first.mpIndividual, *second.mpIndividual, rand);
						rToChange.Mutate(mutationRate, rand);
					}
				}
			}

			return false;
		}

		unsigned SelectCount() const { return (unsigned)(mMaxCount*mSurvivalRate); }

		//Returns the error rate of the best element. In the current implementation
		//selects does not clear the memory (in order to avoid constant reallocations)
		double Select()
		{
			mSelected = true;
			std::stable_sort(mpPopulation, mpPopulation + mMaxCount);
			return mpPopulation[0].mError;
		}

		//Static input parameter means that the Train method will be called always with
		//the same input. Returns the error of the best individual.
		double Train(const AlignedMatrix<Individual::Input, FloatingPoint>& inputMatrix, 
					 const AlignedMatrix<Individual::Output, FloatingPoint>& expectedMatrix, double mutationRate, bool staticInput)
		{
			if (inputMatrix.NumRows() != expectedMatrix.NumRows())
				throw std::string("Different number of rows in the input and expected output marices");
			bool initial = Populate(mutationRate);
			//The first time we evaluate all elements. Beyond that we only evaluate the new ones, if the input is static:
			int startElement = (initial || !staticInput) ? 0 : (int)(mMaxCount*mSurvivalRate);
			Evaluate(inputMatrix, expectedMatrix, startElement);
			return Select();
		}

		void Evaluate(const AlignedMatrix<Individual::Input, FloatingPoint>& inputMatrix, const AlignedMatrix<Individual::Output, FloatingPoint>& expectedMatrix, int skipElements = 0)
		{
			if (inputMatrix.NumRows() != expectedMatrix.NumRows())
				throw std::string("Different number of rows in the input and expected output marices");

			//Preinitialize the temporary matrices: one per OMP thread, to avoid large number of allocations:
			int maxTreads = omp_get_max_threads();
			AlignedMatrix<Individual::Output, FloatingPoint>** pMatrices = new AlignedMatrix<Individual::Output, FloatingPoint>*[maxTreads];
			for (int i = 0; i < maxTreads; ++i)
			{
				pMatrices[i] = new AlignedMatrix<Individual::Output, FloatingPoint>(inputMatrix.NumRows());
			}
			#pragma omp parallel for
			for (int i = skipElements; i < (int)mMaxCount; ++i)
			{
				int threadNum = omp_get_thread_num();
				AlignedMatrix<Individual::Output, FloatingPoint>* pOutputMtrx = pMatrices[omp_get_thread_num()];
				mpPopulation[i].mpIndividual->BatchProcessInputFast(inputMatrix, *pOutputMtrx);
				mpPopulation[i].mError = mpPopulation[i].mpIndividual->CalculateError(*pOutputMtrx, expectedMatrix);
			}
		}

	protected:

	};
}