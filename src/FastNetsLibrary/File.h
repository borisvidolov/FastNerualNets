// Created by Boris Vidolov on 02/15/2014
// Published under Apache 2.0 licence.
#pragma once
#include <string>
#include <sstream>

namespace FastNets
{
/*  Helper class, wrapping FILE* 
	Example:
	{
		File f("foo.txt", "rb");
		fread(..., fp);
	}//Destructor called, calling fclose
*/
class File
{
protected:
	FILE* mpFILE;
private:
	File(const File&){}//No copy, as the destructor will attempt to close the file twice
public:
	File(const char* szFile, const char* szMode)
	{
		mpFILE = NULL;
		errno_t err = fopen_s(&mpFILE, szFile, szMode);
		if (err || !mpFILE)
		{
			std::stringstream stream;
			stream << "Cannot open the file: " << szFile << " ; Error:" << err;
			throw stream.str();
		}
	}

	File(FILE* pFILE):mpFILE(pFILE)
	{
		if (!pFILE)
		{
			throw std::string("Null FILE pointer");
		}
	}

	~File(){ fclose(mpFILE); }

	
	template<class T>
	void WriteOne(const T& t)
	{
		Write(&t, sizeof(t), 1);
	}

	void WriteSize(unsigned size)
	{
		WriteOne<uint32_t>((uint32_t)size);
	}

	template<class T>
	void ReadOne(T& t)
	{
		Read(&t, sizeof(t), 1);
	}

	template<class T>
	void ReadAndVerify(const T& t, const char* szError)
	{
		T temp;
		ReadOne(temp);
		if (temp != t)
			throw std::string(szError);
	}

	void ReadAndVerifySize(unsigned expected, const char* szError)
	{
		ReadAndVerify<uint32_t>((uint32_t)expected, szError);
	}

	template<class T>
	void ReadMany(T* pT, unsigned size)
	{
		Read(pT, sizeof(T), size);
	}

	//Compile-time check that only non-pointer types are used
	template<class T>
	int WriteOne(const T* pT)
	{
		//Make sure it doesn't compile. WriteOne shouldn't be called with pointers
		//The error will be for returning a value
	}

	template<class T>
	void WriteMany(const T* pT, const unsigned numElements)
	{
		Write(pT, sizeof(T), numElements);
	}

	FILE* GetFP(){ return mpFILE; }

protected:
	void Write(const void* buffer, const unsigned elSize, const unsigned numElements)
	{
		int write = fwrite(buffer, elSize, numElements, mpFILE);
		if (numElements != write)
			throw std::string("Unable to write to the file");
	}

	void Read(void* buffer, const unsigned elSize, const unsigned numElements)
	{
		int read = fread(buffer, elSize, numElements, mpFILE);
		if (numElements != read)
			throw std::string("Unable to read from the file");
	}
};//Class File
}//Namespace FastNets