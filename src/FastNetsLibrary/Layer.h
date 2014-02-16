// Created by Boris Vidolov on 02/14/2014
// Published under Apache 2.0 licence.
#pragma once
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <time.h>
#include <sstream>
#include "File.h"

void InitializeOmp();
namespace FastNets
{

/* Represents a single layer in the network. Note that the class will
initialize the OMP threads to achieve maximum performance gain.*/
template<unsigned INPUT, unsigned OUTPUT, class FloatingPoint = double>
class Layer
{
protected:
	_CRT_ALIGN(32) FloatingPoint mWeights[OUTPUT][INPUT];
	_CRT_ALIGN(32) FloatingPoint mReverseWeights[INPUT][OUTPUT];//Cache for faster calculation
	_CRT_ALIGN(32) FloatingPoint mB[OUTPUT];//Input Bias
	_CRT_ALIGN(32) FloatingPoint mC[INPUT];//Output Bias

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
		InitializeOmp();

		boost::mt11213b gen;
		gen.seed((unsigned __int32)time(NULL));
		const int max = 10000;
		boost::random::uniform_int_distribution<> dist(1, max);
		for (int i = 0; i < OUTPUT; ++i)
		{
			for (int j = 0; j < INPUT; ++j)
			{ 
				mWeights[i][j] = GetRandomWeight(dist, gen, max);
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
		InitializeOmp();

		File f(szFile, "rb");
		ReadFromFile(f);
		mReverseWeightsDirty = true;
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
		__int32 fpSize = sizeof(mWeights[0][0]);
		write = fwrite(&fpSize, sizeof(__int32), 1, fp);
		if (1 != write)
			throw std::string("Unable to write to the file");
		write = fwrite(mWeights, sizeof(mWeights[0][0]), INPUT*OUTPUT, fp);
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
		if (1 != read || sizeof(mWeights[0][0]) != fpSize)
			throw std::string("Bad floating point type!");
		read = fread(mWeights, sizeof(mWeights[0][0]), INPUT*OUTPUT, fp);
		if (INPUT*OUTPUT != read)
			throw std::string("Cannot read all of the weights!");
		read = fread(mB, sizeof(mB[0]), OUTPUT, fp);
		if (OUTPUT != read)
			throw std::string("Cannot read all of the input biases!");
		read = fread(mC, sizeof(mC[0]), INPUT, fp);
		if (INPUT != read)
			throw std::string("Cannot read all of the output biases!");
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

