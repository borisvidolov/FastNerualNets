// Created by Boris Vidolov on 02/14/2014
// Published under Apache 2.0 licence.
#pragma once
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <time.h>
#include <sstream>
#include "File.h"
#include "FloatingPoint.h"
#include "Randomizer.h"
#include "AlignedMatrix.h"

namespace FastNets
{

/* Represents a single layer in the network. Note that the class will
initialize the OMP threads to achieve maximum performance gain.*/
template<unsigned INPUT, unsigned OUTPUT, class FloatingPoint = double>
class Layer
{
protected:
	
	AlignedMatrix<INPUT, FloatingPoint>  mWeights;
	AlignedMatrix<OUTPUT, FloatingPoint> mReverseWeights;
	FloatingPoint* mB;//Input Bias
	FloatingPoint* mC;//Output Bias (for reverse calculation)

	bool  mReverseWeightsDirty;
private:
	Layer(const Layer&){}//No copy

/* Public constants */
public:
	const static unsigned Input		= INPUT;
	const static unsigned Output	= OUTPUT;
	typedef typename FloatingPoint FloatingPointType;
/*Constructors and destructors. */
public:

	Layer(bool initialize)
		:mWeights(OUTPUT), mReverseWeights(INPUT)
	{
		Randomizer<> r;

		AllocateMemory();
		if (initialize)
		{
			for (int i = 0; i < OUTPUT; ++i)
			{
				FloatingPoint* pRow = mWeights.GetRow(i);
				for (int j = 0; j < INPUT; ++j)
				{ 
					pRow[j] = GetRandomWeight(r, OUTPUT + 1);
				}
			}
			for (int i = 0; i < OUTPUT; ++i)
				mB[i] = GetRandomWeight(r, OUTPUT + 1);
			for (int i = 0; i < INPUT; ++i)
				mC[i] = GetRandomWeight(r, INPUT + 1);
			mReverseWeightsDirty = true;	
		}
	}

	//Creates a layer by merging the two:
	Layer(const Layer& merge1, const Layer& merge2, Randomizer<>& r)
		:mWeights(OUTPUT), mReverseWeights(INPUT)
	{
		AllocateMemory();
		Merge(merge1, merge2, r);
	}

	//Creates a random merge of the two parents. Used in genetic algorithms
	void SetFromMergedParents(const Layer& merge1, const Layer& merge2, Randomizer<>& r)
	{
		Merge(merge1, merge2, r);
	}


	~Layer()
	{
		_aligned_free(mB);
		_aligned_free(mC);
	}

/*Public methods */
public:
	void WriteToFile(const char* szFile)
	{
		File f(szFile, "wb");
		WriteToFile(f);
	}

	void WriteToFile(File& rFile)
	{
		rFile.WriteSize(INPUT);
		rFile.WriteSize(OUTPUT);
		rFile.WriteSize(sizeof(FloatingPointType));

		mWeights.WriteToFile(rFile);
		rFile.WriteMany(mB, OUTPUT);
		rFile.WriteMany(mC, INPUT);
	}

	void ReadFromFile(const char* szFile)
	{
		File f(szFile, "rb");
		ReadFromFile(f);
	}

	void ReadFromFile(File& rFile)
	{
		rFile.ReadAndVerifySize(INPUT, "Wrong input size");
		rFile.ReadAndVerifySize(OUTPUT, "Wrong output size");
		rFile.ReadAndVerifySize(sizeof(FloatingPointType), "Wrong floating point file");

		mWeights.ReadFromFile(rFile);
		rFile.ReadMany(mB, OUTPUT);
		rFile.ReadMany(mC, INPUT);
		mReverseWeightsDirty = true;
	}

	bool IsSame(const Layer& other) const
	{
		for (unsigned i = 0; i < OUTPUT; ++i)
		{
			if (!AreSame<FloatingPoint>(mWeights.GetRow(i), other.mWeights.GetRow(i), INPUT))
				return false;
		}
		if (!AreSame<FloatingPoint>((FloatingPoint*)mB, (FloatingPoint*)other.mB, OUTPUT))
			return false;
		if (!AreSame<FloatingPoint>((FloatingPoint*)mC, (FloatingPoint*)other.mC, INPUT))
			return false;
		return true;
	}

	void ProcessInputSlow(const FloatingPoint* input, FloatingPoint* output) const
	{
		for (unsigned i = 0; i < OUTPUT; ++i)
		{
			FloatingPoint accum = mB[i];
			const FloatingPoint* pt = mWeights.GetRow(i);
			for (unsigned j = 0; j < INPUT; ++j)
			{ 
				accum += (*(pt++))*input[j];
			}
			output[i] = OutputFunction(accum);
		}	
	}

	/*IMPORTANT: This one requires _CRT_ALIGN(32) pointers */
	void ProcessInputFast(const FloatingPoint* input, FloatingPoint* output) const
	{
		ProcessInputAVX(input, output, INPUT, OUTPUT, mWeights.GetBuffer(), mB);
	}

	void Mutate(FloatingPoint rate, Randomizer<>& r)
	{
		for (unsigned i = 0; i < OUTPUT; ++i)
		{
			MutateWeight(mB[i], rate, r);
			FloatingPoint* pt = mWeights.GetRow(i);
			for (unsigned j = 0; j < INPUT; ++j)
			{ 
				MutateWeight(*pt, rate, r);
				++pt;
			}
		}	
	}

	void CalculateBackPropagationError(const FloatingPointType* lowerOutput, const FloatingPointType* upperError, FloatingPointType* lowerError) const
	{
		throw std::string("Implement me");
	}



/* Internal implementaiton */
protected:

	void UpdateReverseWeights()
	{
		if (!mReverseWeightsDirty)
			return;
		throw std::string("Implement me");
	}
	//Compile-time checks on the parameters
	void ValidateTemplateParameters();

	double GetRandomWeight(Randomizer<>& rand, double divider)
	{
		double value = rand.RangeNext(6);
		if (value < 0.0001 && value > -0.0001)
		{
			value = _copysign(0.001, value);
		}
		value /= divider;
		return value;
	}

	void AllocateMemory()
	{
		mB = (FloatingPoint*)_aligned_malloc(OUTPUT*sizeof(FloatingPoint), 32);
		mC = (FloatingPoint*)_aligned_malloc(INPUT*sizeof(FloatingPoint), 32);
	}

	//Changes the "source" weight with the specified rate:
	void MutateWeight(FloatingPoint& source, double rate, Randomizer<>& rand)
	{
        if (source < 0.00001 && source > -0.00001)
        {
            //Abillity to bring back really small numbers:
			source = -_copysign(0.001, source);//Pass to the other side of the 0
		}

		double quotient = rand.OffsetNext(rate);

        source = source*quotient;
	}

	void Merge(const Layer& layer1, const Layer& layer2, Randomizer<>& rand)
	{
		FloatingPoint *pt;
		const FloatingPoint *pt1, *pt2;
		for (unsigned i = 0; i < OUTPUT; ++i)
		{
			mB[i] = rand.NextBool() ? layer1.mB[i] : layer2.mB[i];
			pt = mWeights.GetRow(i);
			pt1 = layer1.mWeights.GetRow(i);
			pt2 = layer2.mWeights.GetRow(i);
			for (unsigned j = 0; j < INPUT; ++j)
			{ 
				*pt = rand.NextBool() ? *pt1 : *pt2;
				++pt;
				++pt1;
				++pt2;
			}
		}	
	}
};//Layer class

}//FastNets namespace

