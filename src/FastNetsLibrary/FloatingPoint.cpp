// Created by Boris Vidolov on 02/15/2014
// Published under Apache 2.0 licence

#include "stdafx.h"
#include "FloatingPoint.h"

//TODO: Add non-inline calculation methods here

namespace FastNets
{

void ProcessInputAVX(double* input, double* output, unsigned inputSize, unsigned outputSize, double* weights, double* bias)
{
	unsigned alignedInputSize = AVXAlign(inputSize);
	for (int i = 0; i < (int)outputSize; ++i)
	{
		double* pWeights = &weights[i*alignedInputSize];
		double* pInput = input;
		unsigned size8 = inputSize/8;
		register __m256d res = _mm256_set_pd(bias[i], 0, 0, 0);

		for (unsigned j = 0; j < size8; ++j)
		{
			register __m256d m4d1 = _mm256_load_pd(pInput);
			register __m256d m4d2 = _mm256_load_pd(pInput + 4);
			register __m256d weights1 = _mm256_load_pd(pWeights);
			register __m256d weights2 = _mm256_load_pd(&pWeights[4]);
			m4d1 = _mm256_mul_pd(m4d1, weights1);
			m4d2 = _mm256_mul_pd(m4d2, weights2);
			res = _mm256_add_pd(res, m4d1);
			res = _mm256_add_pd(res, m4d2);
			pWeights += 8;
			pInput += 8;
		}

		res = _mm256_hadd_pd(res, res);//{r0 + r1, r0 + r1, r2 + r3, r2 + r3}
		register __m256d tmp = _mm256_permute2f128_pd(res, res, 1);//{r2 + r3, r2 + r3, r0 + r1, r0 + r1}
		res = _mm256_add_pd(res, tmp);//{r0 + r1 + r2 + r3, ....}
		double result = res.m256d_f64[0];
		//Now deal with the rest:
		unsigned reminder = inputSize % 8;
		for (unsigned j = 0; j < reminder; ++j)
		{
			result += (*(pWeights++))*(*(pInput++));
		}
		output[i] = OutputFunction(result);
	}
}

void TransferAlignedInput(const double* pInput, unsigned inputFeatures, unsigned numInputs, double* avxAlignedOutput)
{
	throw std::string("Implement me");
}


}
