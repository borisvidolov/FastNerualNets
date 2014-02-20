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

namespace FastNets
{

/* Represents a single layer in the network. Note that the class will
initialize the OMP threads to achieve maximum performance gain.*/
template<unsigned INPUT, unsigned OUTPUT, class FloatingPoint = double>
class __declspec(align(32)) Layer
{
protected:
	FloatingPoint* mWeights;
	FloatingPoint* mReverseWeights;//Cache for faster calculation
	FloatingPoint* mB;//Input Bias
	FloatingPoint* mC;//Output Bias (for reverse calculation)

	bool  mReverseWeightsDirty;
private:
	Layer(const Layer&){}//No copy

/* Public constants */
public:
	const static unsigned Input		= INPUT;
	const static unsigned ALIGNED_INPUT = (INPUT + 31) & ~31;
	const static unsigned Output	= OUTPUT;
	const static unsigned ALIGNED_OUTPUT = (OUTPUT + 31) & ~31;
	typedef typename FloatingPoint FloatingPointType;
/*Constructors and destructors. */
public:

	Layer(void)
	{
		Randomizer<> r;

		AllocateMemory();

		for (int i = 0; i < OUTPUT; ++i)
		{
			for (int j = 0; j < INPUT; ++j)
			{ 
				mWeights[i*ALIGNED_INPUT + j] = GetRandomWeight(r);
			}
		}
		for (int i = 0; i < OUTPUT; ++i)
			mB[i] = GetRandomWeight(r);
		for (int i = 0; i < INPUT; ++i)
			mC[i] = GetRandomWeight(r);
		mReverseWeightsDirty = true;	
	}

	//Reads from file:
	Layer(const char* szFile)
	{
		AllocateMemory();

		File f(szFile, "rb");
		ReadFromFile(f);
		mReverseWeightsDirty = true;
	}

	~Layer()
	{
		_aligned_free(mWeights);
		_aligned_free(mReverseWeights);
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

	void WriteToFile(FILE* fp)
	{
		__int32 input = INPUT;
		__int32 output = OUTPUT;
		int write = fwrite(&input, sizeof(input), 1, fp);
		if (1 != write)
			throw std::string("Unable to write to the file");
		write = fwrite(&output, sizeof(output), 1, fp);
		if (1 != write)
			throw std::string("Unable to write to the file");
		__int32 fpSize = sizeof(mWeights[0]);
		write = fwrite(&fpSize, sizeof(__int32), 1, fp);
		if (1 != write)
			throw std::string("Unable to write to the file");
		write = fwrite(mWeights, sizeof(mWeights[0]), ALIGNED_INPUT*OUTPUT, fp);
		if (ALIGNED_INPUT*OUTPUT != write)
			throw std::string("Unable to write to the file");
		write = fwrite(mB, sizeof(mB[0]), OUTPUT, fp);
		if (OUTPUT != write)
			throw std::string("Unable to write to the file");
		write = fwrite(mC, sizeof(mC[0]), INPUT, fp);
		if (INPUT != write)
			throw std::string("Unable to write to the file");	
	}

	void ReadFromFile(FILE* fp)
	{
		__int32 input = 0;
		__int32 output = 0;
		int read = fread(&input, sizeof(input), 1, fp);
		if (1 != read || INPUT != input)
			throw std::string("Bad input size!");
		read = fread(&output, sizeof(output), 1, fp);
		if (1 !=  read || OUTPUT != output)
			throw std::string("Bad output size!");
		__int32 fpSize = 0;
		read = fread(&fpSize, sizeof(__int32), 1, fp);
		if (1 != read || sizeof(mWeights[0]) != fpSize)
			throw std::string("Bad floating point type!");
		read = fread(mWeights, sizeof(mWeights[0]), ALIGNED_INPUT*OUTPUT, fp);
		if (ALIGNED_INPUT*OUTPUT != read)
			throw std::string("Cannot read all of the weights!");
		read = fread(mB, sizeof(mB[0]), OUTPUT, fp);
		if (OUTPUT != read)
			throw std::string("Cannot read all of the input biases!");
		read = fread(mC, sizeof(mC[0]), INPUT, fp);
		if (INPUT != read)
			throw std::string("Cannot read all of the output biases!");
	}

	bool IsSame(const Layer& other) const
	{
		if (!AreSame<FloatingPoint>((FloatingPoint*)mWeights, (FloatingPoint*)other.mWeights, INPUT*OUTPUT))
			return false;
		if (!AreSame<FloatingPoint>((FloatingPoint*)mB, (FloatingPoint*)other.mB, OUTPUT))
			return false;
		if (!AreSame<FloatingPoint>((FloatingPoint*)mC, (FloatingPoint*)other.mC, INPUT))
			return false;
		return true;
	}

	void ProcessInputSlow(FloatingPoint* input, FloatingPoint* output)
	{
		FloatingPoint* pt = mWeights;
		for (unsigned i = 0; i < OUTPUT; ++i)
		{
			FloatingPoint accum = mB[i];
			pt = mWeights + ALIGNED_INPUT*i;
			for (unsigned j = 0; j < INPUT; ++j)
			{ 
				accum += (*(pt++))*input[j];
			}
			output[i] = OutputFunction(accum);
		}	
	}

	/*IMPORTANT: This one requires _CRT_ALIGN(32) pointers */
	void ProcessInputFast(FloatingPoint* input, FloatingPoint* output)
	{
		ProcessInputAVX(input, output, INPUT, OUTPUT, mWeights, mB);
	}

	void Mutate(FloatingPoint rate)
	{
		Randomizer<> r;
		FloatingPoint* pt = mWeights;
		for (unsigned i = 0; i < OUTPUT; ++i)
		{
			MutateWeight(mB[i], rate, r);
			pt = mWeights + ALIGNED_INPUT*i;
			for (unsigned j = 0; j < INPUT; ++j)
			{ 
				MutateWeight(*pt);
				++pt;
			}
		}	
	}

	static Layer* Merge(Layer* net1, Layer* net2)
	{
		throw std::string("Implement me");
	}

/* Internal implementaiton */
protected:
	//Compile-time checks on the parameters
	void ValidateTemplateParameters();

	double GetRandomWeight(Randomizer<>& rand)
	{
		double value = rand.RangeNext(6);
		if (value < 0.0001 && value > -0.0001)
		{
			value = _copysign(0.001, value);
		}
		return value;// /Input;
	}

	void AllocateMemory()
	{
		mWeights = (FloatingPoint*)_aligned_malloc(ALIGNED_INPUT*OUTPUT*sizeof(FloatingPoint), 32);
		mReverseWeights = (FloatingPoint*)_aligned_malloc(INPUT*ALIGNED_OUTPUT*sizeof(FloatingPoint), 32);
		mB = (FloatingPoint*)_aligned_malloc(OUTPUT*sizeof(FloatingPoint), 32);
		mC = (FloatingPoint*)_aligned_malloc(INPUT*sizeof(FloatingPoint), 32);
	}

	//Changes the "source" weight with the specified rate:
	void MutateWeight(FloatingPoint& source, double rate, Randomizer<>& rand)
	{
        if (source < 0.00001 && source > -0.00001)
        {
            //Abillity to bring back really small numbers:
			value = -_copysign(0.001, source);//Pass to the other side of the 0
		}

		double quotient = rand.OffsetNext;

        source = source*quotient;
	}
};//Layer class

}//FastNets namespace

