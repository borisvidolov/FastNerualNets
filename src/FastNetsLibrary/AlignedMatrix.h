// Created by Boris Vidolov on 02/27/2014
// Published under Apache 2.0 licence.
#pragma once
#include <malloc.h>
#include "FloatingPoint.h"

namespace FastNets
{

//A helper class to keep matrix, so that each row is at aligned memory
//that can be used in AVX or other SIMD instructions. The current implementation aligns to 32 bytes
//"ALIGNMENT" is in bytes.
//TODO: add the alignment as a parameter, if needed.
template<unsigned ROWSIZE, class FloatingPointType = double>
class AlignedMattrix
{
public:
	FloatingPointType* mpMatrix;
	unsigned		   mNumRows;
	unsigned AlignedRowSize = AVXAlignType(ROWSIZE, sizeof(FloatingPointType));
private:
	AlignedMattrix(const AlignedMattrix&){}//No copy
public:
	//Allocates the storage:
	AlignedMattrix(unsigned rowCount)
		:mNumRows(rowCount)
	{
		AllocateBuffer();
	}

	AlignedMattrix(FloatingPointType* pNonAlignedBuffer, unsigned rowCount)
		:mNumRows(rowCount)
	{
		AllocateBuffer();
		TransferAlignedInput(pNonAlignedBuffer, mpMatrix, mNumRows, mpMatrix);
	}

	~AlignedMattrix()
	{
		FreeBuffer();
	}

	unsigned GetRowByteIndex(unsigned row) const 
	{
		if (row > mNumRows)
			throw std::string("Row out of range.");
		return row*AlignedRowSize;
	}
	FloatingPointType* GetRow(unsigned row){ return mpMatrix[GetRowByteIndex(row)]; }
	const FloatingPointType* GetRow(unsigned row) const { return mpMatrix[GetRowByteIndex(row)]; }

	unsigned NumRows() const { return mNumRows; }
	FloatingPointType* GetBuffer() { return mpMatrix; }
	const FloatingPointType* GetBuffer() const { return mpMatrix; }

	void AllocateBuffer()
	{
		mpMatrix = (FloatingPointType*)_aligned_malloc(mNumRows*AlignedRowSize*sizeof(FloatingPointType));
	}

	void FreeBuffer()
	{
		if (mpMatrix)
			_aligned_free(mpMatrix);
	}
};

}