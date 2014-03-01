// Created by Boris Vidolov on 02/27/2014
// Published under Apache 2.0 licence.
#pragma once
#include <malloc.h>
#include "FloatingPoint.h"
#include "File.h"

namespace FastNets
{

//A helper class to keep matrix, so that each row is at aligned memory
//that can be used in AVX or other SIMD instructions. The current implementation aligns to 32 bytes
//"ALIGNMENT" is in bytes.
//TODO: add the alignment as a parameter, if needed.
template<unsigned ROWSIZE, class FloatingPointType = double>
class AlignedMatrix
{
public:
	FloatingPointType* mpMatrix;
	unsigned		   mNumRows;
	const static unsigned AlignedRowSize = AVXAlignType(ROWSIZE, sizeof(FloatingPointType));
private:
	AlignedMatrix(const AlignedMatrix&){}//No copy
public:
	//Allocates the storage:
	AlignedMatrix(unsigned rowCount)
		:mNumRows(rowCount)
	{
		AllocateBuffer();
	}

	AlignedMatrix(const FloatingPointType* pNonAlignedBuffer, unsigned rowCount)
		:mNumRows(rowCount)
	{
		AllocateBuffer();
		TransferAlignedInput(pNonAlignedBuffer, ROWSIZE, mNumRows, mpMatrix);
	}

	~AlignedMatrix()
	{
		FreeBuffer();
	}

	void WriteToFile(File& rFile)
	{
		rFile.WriteSize(mNumRows);
		rFile.WriteSize(sizeof(FloatingPointType));
		rFile.WriteSize(ROWSIZE);
		for (unsigned i = 0; i < mNumRows; ++i)
		{
			rFile.WriteMany(GetRow(i), ROWSIZE);
		}
	}

	//read the contents of already constructed matrix:
	void ReadFromFile(File& rFile)
	{
		rFile.ReadAndVerifySize(mNumRows, "Wrong number of rows");
		rFile.ReadAndVerifySize(sizeof(FloatingPointType), "Wrong floating point type");
		rFile.ReadAndVerifySize(ROWSIZE, "Wrong row size");
		for (unsigned i = 0; i < mNumRows; ++i)
		{
			rFile.ReadMany(GetRow(i), ROWSIZE);
		}
	}

	bool IsSame(const AlignedMatrix& other) const
	{
		for (unsigned i = 0; i < mNumRows; ++i)
		{
			if (!AreSame(GetRow(i), other.GetRow(i), ROWSIZE))
				return false;
		}
		return true;
	}

	unsigned GetRowByteIndex(unsigned row) const 
	{
		if (row > mNumRows)
			throw std::string("Row out of range.");
		return row*AlignedRowSize;
	}
	FloatingPointType* GetRow(unsigned row){ return mpMatrix + GetRowByteIndex(row); }
	const FloatingPointType* GetRow(unsigned row) const { return mpMatrix + GetRowByteIndex(row); }

	unsigned NumRows() const { return mNumRows; }
	FloatingPointType* GetBuffer() { return mpMatrix; }
	const FloatingPointType* GetBuffer() const { return mpMatrix; }

	void AllocateBuffer()
	{
		mpMatrix = (FloatingPointType*)_aligned_malloc(mNumRows*AlignedRowSize*sizeof(FloatingPointType), 32);
	}

	void FreeBuffer()
	{
		if (mpMatrix)
			_aligned_free(mpMatrix);
	}
};

}