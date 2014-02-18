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

	operator FILE* (){ return mpFILE; }
};//Class File
}//Namespace FastNets