// Created by Boris Vidolov on 02/15/2014
// Published under Apache 2.0 licence.
#pragma once
#include "Layer.h"
#include "File.h"

namespace FastNets
{
//This class implements a deep network through stacking layers.
//Example: 
//	Net<5, Net<3, Net<1>>> n;
//The line above creates a network with 3 layers, 5 input neurons, 3 hidden ones and 1 output.
//It is possible to stack this way to arbitrary depth:
//	Net<10, Net<9, Net<8, Net<7, Net<6, Net<5>>>>>> n5;

template<unsigned INPUT, class UpperNet = double>
class Net
{
/* Public constants */
public:
	const static unsigned Input = INPUT;
	const static unsigned Output = UpperNet::Output;
	const static bool	  Last = false;

	typedef typename UpperNet::FloatingPointType FloatingPointType;
protected:
	//Unfortunately, the stack size is limited and insufficient for really deep
	//networks. So we dynamically allocate the large data here:
	Layer<INPUT, UpperNet::Input, FloatingPointType>		mInputLayer;
	UpperNet									  			mNext;
private:
	Net(const Net&){}//No copy

/*Constructors and destructors */
public:
	Net():mInputLayer(true){}

	Net(const char* szFile):mInputLayer(false)
	{
		File f(szFile, "rb");
		ReadFromFile(f);
	}

	Net(const Net& first, const Net& second, Randomizer<>& rand)
		:mInputLayer(first.mInputLayer, second.mInputLayer, rand),
		mNext(first.mNext, second.mNext, rand)
	{
	}

	//Creates a random merge of the two parents. Used in genetic algorithms
	void SetFromMergedParents(const Net& first, const Net& second, Randomizer<>& rand)
	{
		mInputLayer.SetFromMergedParents(first.mInputLayer, second.mInputLayer, rand);
		mNext.SetFromMergedParents(first.mNext, second.mNext, rand);
	}

	void ReadFromFile(File& rFile)
	{
		mInputLayer.ReadFromFile(rFile);
		mNext.ReadFromFile(rFile);
	}

	//Writes to the file. Assumes that the file pointer
	//current location is already set to the right place:
	void WriteToFile(const char* szFile)
	{
		File f(szFile, "wb");
		WriteToFile(f);
	}

	void WriteToFile(File& rFile)
	{
		mInputLayer.WriteToFile(rFile);
		mNext.WriteToFile(rFile);
	}

	bool IsSame(const Net& other) const
	{
		return mInputLayer.IsSame(other.mInputLayer) && mNext.IsSame(other.mNext);
	}

	/* Processes a matrix, which rows are inputs to a matrix, which rows are the outputs. 
	"count" specifies the number of rows. */
	void BatchProcessInputSlow(const AlignedMatrix<INPUT, FloatingPointType>& input, AlignedMatrix<Output, FloatingPointType>& output) const
	{
		if (input.NumRows() != output.NumRows())
			throw std::string("Different number of rows between the input and the output");
		for (unsigned i = 0; i < input.NumRows(); ++i)
		{
			ProcessInputSlow(input.GetRow(i), output.GetRow(i));
		}
	}

	double CalculateError(const AlignedMatrix<Output, FloatingPointType>& output, const AlignedMatrix<Output, FloatingPointType>& expected) const
	{
		//This code can be optimized with AVX, OMP, etc. However, at this point
		//the code is only 0.6% of the execution time. It is worth optimizing only for very small networks
		double accum = 0;
		if (output.NumRows() != expected.NumRows())
			throw std::string("Different number of rows between the input and the output");
		for (unsigned i = 0; i < output.NumRows(); ++i)
		{
			double localSquares = 0;
			const FloatingPointType* rowOutput = output.GetRow(i);
			const FloatingPointType* rowExpected = expected.GetRow(i);
			for (unsigned j = 0; j < Output; ++j)
			{
				double delta = rowOutput[j] - rowExpected[j];
				localSquares += delta*delta;
			}
			localSquares /= Output;
			accum += localSquares;
		}
		accum /= output.NumRows();
		return accum;
	}

	/* Forward calculation of the network. The method uses the first INPUT elements
	   of the "input" array, so the array will need to have at least as much elements.
	   The same applies to the "output" array, where the last layer of the net will
	   put its data there.*/
	void ProcessInputSlow(const FloatingPointType* input, FloatingPointType* output) const
	{
		if (UpperNet::Last)//Should be constant expression
		{
			//The next layer is dummy, it doesn't process. Just put the result in the "output"
			mInputLayer.ProcessInputSlow(input, output);
		}
		else
		{
			FloatingPointType intermediate[UpperNet::Input];
			mInputLayer.ProcessInputSlow(input, intermediate);
			mNext.ProcessInputSlow(intermediate, output);
		}
	}

	/* Processes a matrix, which rows are inputs to a matrix, which rows are the outputs. 
	"count" specifies the number of rows. */
	void BatchProcessInputFast(const AlignedMatrix<INPUT, FloatingPointType>& input, AlignedMatrix<Output, FloatingPointType>& output) const
	{
		if (input.NumRows() != output.NumRows())
			throw std::string("Different number of rows between the input and the output");
		#pragma omp parallel for
		for (int i = 0; i < (int)input.NumRows(); ++i)
		{
			ProcessInputFast(input.GetRow(i), output.GetRow(i));
		}
	}

	/* Forward calculation of the network. The method uses the first INPUT elements
	   of the "input" array, so the array will need to have at least as much elements.
	   The same applies to the "output" array, where the last layer of the net will
	   put its data there.*/
	void ProcessInputFast(const FloatingPointType* input, FloatingPointType* output) const
	{
		if (UpperNet::Last)//Should be constant expression
		{
			//The next layer is dummy, it doesn't process. Just put the result in the "output"
			mInputLayer.ProcessInputFast(input, output);
		}
		else
		{
			_CRT_ALIGN(32) FloatingPointType intermediate[UpperNet::Input];
			mInputLayer.ProcessInputFast(input, intermediate);
			mNext.ProcessInputFast(intermediate, output);
		}
	}

	void Mutate(double rate)
	{
		Randomizer<> rand;
		Mutate(rate, rand);
	}

	void Mutate(double rate, Randomizer<>& rand)
	{
		mInputLayer.Mutate(rate, rand);
		mNext.Mutate(rate, rand);
	}

	double BackPropagation(const AlignedMatrix<INPUT, FloatingPointType>& output, const AlignedMatrix<Output, FloatingPointType>& expected, double learningRate) const
	{
		throw std::string("Implement me");
	}

private:
	double BackPropagation(const FloatingPointType* input, FloatingPointType* output, const FloatingPointType* expected, 
								 FloatingPointType* errors, double learningRate, bool firstLayer)
	{
		_CRT_ALIGN(32) FloatingPointType intermediate[UpperNet::Input];
		_CRT_ALIGN(32) FloatingPointType localError[UpperNet::Input];
		//Forward pass:
		mInputLayer.ProcessInputFast(input, intermediate);
		double outputError = mNext.BackPropagation(intermediate, output, expected, localError, learningRate, false);
		//Backward pass:
		//Calculate the errors for the lower level, unless there is no lower one:
		if (!firstError)
		{
			mInputLayer.CalculateBackPropagationError(errors, localError);
		}
		mInputLayer.UpdateWeights(localError, learningRate);
		return outputError;
	}
};//Net class

//Specialization for the ending, most implementation is empty
template<unsigned INPUT>
class Net<INPUT, double>
{
/* Public constants */
public:
	const static unsigned Input = INPUT;
	const static unsigned Output = INPUT;
	const static bool Last = true;//Identifies the last (dummy) layer.

	typedef double FloatingPointType;
public:
	Net(){}
	Net(const char* szFile){}      
	Net(const Net& first, const Net& second, Randomizer<>& rand){}
	void WriteToFile(const char* szFile){}
	void WriteToFile(File& rFile){}
	void ReadFromFile(File& rFile){}
	bool IsSame(const Net& other) const { return true; }
	double ProcessInputSlow(const FloatingPointType* input, FloatingPointType* output) const { throw std::string("Execution Flow error"); }
	double ProcessInputFast(const FloatingPointType* input, FloatingPointType* output) const { throw std::string("Execution Flow error"); }
	void Mutate(double rate, Randomizer<>& rand){}
	//Creates a random merge of the two parents. Used in genetic algorithms
	void SetFromMergedParents(const Net& first, const Net& second, Randomizer<>& rand){}
	double BackPropagation(const FloatingPointType* input, FloatingPointType* output, const FloatingPointType* expected, 
								 FloatingPointType* errors, double learningRate, bool firstLayer)
	{
		throw std::string("Impelement me");
	}
};

}//FastNets namespace
