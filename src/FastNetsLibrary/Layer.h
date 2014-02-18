// Created by Boris Vidolov on 02/14/2014
// Published under Apache 2.0 licence.
#pragma once
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <time.h>
#include <sstream>
#include "File.h"
#include "FloatingPoint.h"

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
	const static unsigned Output	= OUTPUT;
	typedef typename FloatingPoint FloatingPointType;
/*Constructors and destructors. */
public:

	Layer(void)
	{
		ValidateTemplateParameters();

		boost::mt11213b gen;
		gen.seed((unsigned __int32)time(NULL));
		const int max = 10000;
		boost::random::uniform_int_distribution<> dist(1, max);

		mWeights = (double*)_aligned_malloc(INPUT*OUTPUT*sizeof(double), 32);
		mReverseWeights = (double*)_aligned_malloc(INPUT*OUTPUT*sizeof(double), 32);
		mB = (double*)_aligned_malloc(OUTPUT*sizeof(double), 32);
		mC = (double*)_aligned_malloc(INPUT*sizeof(double), 32);

		for (int i = 0; i < OUTPUT; ++i)
		{
			for (int j = 0; j < INPUT; ++j)
			{ 
				mWeights[i*INPUT + j] = GetRandomWeight(dist, gen, max);
			}
		}
		for (int i = 0; i < OUTPUT; ++i)
			mB[i] = GetRandomWeight(dist, gen, max);
		for (int i = 0; i < INPUT; ++i)
			mC[i] = GetRandomWeight(dist, gen, max);
		mReverseWeightsDirty = true;	
	}

	//Reads from file:
	Layer(const char* szFile)
	{
		ValidateTemplateParameters();

		mWeights = (double*)_aligned_malloc(INPUT*OUTPUT*sizeof(double), 32);
		mReverseWeights = (double*)_aligned_malloc(INPUT*OUTPUT*sizeof(double), 32);
		mB = (double*)_aligned_malloc(OUTPUT*sizeof(double), 32);
		mC = (double*)_aligned_malloc(INPUT*sizeof(double), 32);

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
		write = fwrite(mWeights, sizeof(mWeights[0]), INPUT*OUTPUT, fp);
		if (INPUT*OUTPUT != write)
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
		read = fread(mWeights, sizeof(mWeights[0]), INPUT*OUTPUT, fp);
		if (INPUT*OUTPUT != read)
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
		for (int i = 0; i < OUTPUT; ++i)
		{
			double* pt = &mWeights[i*INPUT];
			unsigned size8 = INPUT/8;
			register __m256d res = _mm256_set_pd(mB[i], 0, 0, 0);

			for (unsigned j = 0; j < size8; ++j)
			{
				register __m256d m4d1 = _mm256_load_pd(&input[8*j]);
				register __m256d m4d2 = _mm256_load_pd(&input[8*j + 4]);
				register __m256d weights1 = _mm256_load_pd(pt);
				register __m256d weights2 = _mm256_load_pd(&pt[4]);
				m4d1 = _mm256_mul_pd(m4d1, weights1);
				m4d2 = _mm256_mul_pd(m4d2, weights2);
				res = _mm256_add_pd(res, m4d1);
				res = _mm256_add_pd(res, m4d2);
				pt += 8;
			}
			res = _mm256_hadd_pd(res, res);//{r0 + r1, r0 + r1, r2 + r3, r2 + r3}
			register __m256d tmp = _mm256_permute2f128_pd(res, res, 1);//{r2 + r3, r2 + r3, r0 + r1, r0 + r1}
			res = _mm256_add_pd(res, tmp);//{r0 + r1 + r2 + r3, ....}
			output[i] = OutputFunction(res.m256d_f64[0]);
		}
	}

/* Internal implementaiton */
protected:
	//Compile-time checks on the parameters
	void ValidateTemplateParameters();

	double GetRandomWeight(boost::random::uniform_int_distribution<>& dist, boost::mt11213b& gen, const int max )
	{
		double value = 12*((double)dist(gen)/max) - 6;
		if (abs(value) < 0.001)//Don't create links with 0 strength
			value = 0.1;	
		return value;// /Input;
	}
};//Layer class

#pragma warning (push)
#pragma warning (disable:4101)
template<unsigned INPUT, unsigned OUTPUT, class FloatingPoint>
void Layer<INPUT, OUTPUT, FloatingPoint>::ValidateTemplateParameters()
{
	//Compile-time checks to ensure that the parameters meet the expectations
	int x[!(INPUT % 8)];//Ensure division by 8
	int y[!(OUTPUT % 8)];//Ensure division by 8
}
#pragma warning (pop)

}//FastNets namespace

