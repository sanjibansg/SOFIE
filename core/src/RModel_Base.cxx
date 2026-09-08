#include <limits>
#include <algorithm>
#include <cctype>
#include "SOFIE/RModel_Base.hxx"


namespace SOFIE {

RModel_Base::RModel_Base(std::string name, std::string parsedtime):fFileName(name), fParseTime(parsedtime) {
    fName = fFileName.substr(0, fFileName.rfind("."));
    fName = UTILITY::Clean_name(fName);
}

void RModel_Base::GenerateHeaderInfo(std::string& hgname) {
    fGC += ("//Code generated automatically by TMVA for Inference of Model file [" + fFileName + "] at [" + fParseTime.substr(0, fParseTime.length()-1) +"] \n");
    // add header guards
    hgname = fName;
    std::transform(hgname.begin(), hgname.end(), hgname.begin(), [](unsigned char c) {
                       return std::toupper(c);
                   } );
    hgname = "SOFIE_" + hgname + "_HXX";
    fGC += "\n#ifndef " + hgname + "\n";
    fGC += "#define " + hgname + "\n\n";
    for (auto& i: fNeededStdLib) {
        fGC += "#include <" + i + ">\n";
    }
    for (auto& i: fCustomOpHeaders) {
        fGC += "#include \"" + i + "\"\n";
    }
    // for the session we need to include SOFIE_Common functions
    //needed for convolution operator (need to add a flag)
    fGC += "#include \"SOFIE/SOFIE_common.hxx\"\n";
    if (fUseWeightFile)
        fGC += "#include <fstream>\n";

    if (fWeightFile == WeightFileType::RootBinary){
    #ifdef SOFIE_SUPPORT_ROOT_BINARY
        // Include TFile when saving the weights in a binary ROOT file
            fGC += "#include \"TFile.h\"\n";
    #else
        throw std::runtime_error("sofie: ROOT binary weight file option is enabled but the code is not compiled with ROOT support");
    #endif
    
    }

    fGC += "\nnamespace SOFIE_" + fName + "{\n";
    if (!fNeededBlasRoutines.empty()) {
        fGC += ("namespace BLAS{\n");
        for (auto &routine : fNeededBlasRoutines) {
            if (routine == "Gemm") {
                fGC += ("\textern \"C\" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,\n"
                        "\t                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,\n"
                        "\t                       const float * beta, float * C, const int * ldc);\n");
            } else if (routine == "Gemv") {
                fGC += ("\textern \"C\" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,\n"
                        "\t                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);\n");
            } else if (routine == "Axpy") {
                fGC += ("\textern \"C\" void saxpy_(const int * n, const float * alpha, const float * x,\n"
                        "\t                         const int * incx, float * y, const int * incy);\n");
            } else if (routine == "Copy") {
                fGC += ("\textern \"C\" void scopy_(const int *n, const float* x, const int *incx, float* y, const int* incy);\n");
            }
        }
        fGC += ("}//BLAS\n");
    }
}

void RModel_Base::GenerateHeaderInfo_GPU_ALPAKA(std::string& hgname) {
    fGC += ("//Code generated automatically by TMVA for GPU Inference using ALPAKA of Model file [" + fFileName + "] at [" + fParseTime.substr(0, fParseTime.length()-1) +"] \n");
    // add header guards
    hgname = fName;
    std::transform(hgname.begin(), hgname.end(), hgname.begin(), [](unsigned char c) {
                       return std::toupper(c);
                   } );
    hgname = "SOFIE_" + hgname + "_HXX";
    fGC += "\n#ifndef " + hgname + "\n";
    fGC += "#define " + hgname + "\n\n";
    for (auto& i: fNeededStdLib) {
        fGC += "#include <" + i + ">\n";
    }
    for (auto& i: fCustomOpHeaders) {
        fGC += "#include \"" + i + "\"\n";
    }
    fGC += "#include <alpaka/alpaka.hpp>\n";
    fGC += "#include <sofieBLAS/sofieBLAS.hpp>\n";
    fGC += "#include <span>\n";

    // for the session we need to include SOFIE_Common functions
    //needed for convolution operator (need to add a flag)
    fGC += "#include \"SOFIE/SOFIE_common.hxx\"\n";
    if (fUseWeightFile)
        fGC += "#include <fstream>\n";

    if (fWeightFile == WeightFileType::RootBinary){
        #ifdef SOFIE_SUPPORT_ROOT_BINARY
            // Include TFile when saving the weights in a binary ROOT file
                fGC += "#include \"TFile.h\"\n";
        #else 
            throw std::runtime_error("sofie: ROOT binary weight file option is enabled but the code is not compiled with ROOT support");
        #endif
    }

    fGC += "\n#ifndef SOFIE_ALPAKA_DIM1D_DEFINED\n";
    fGC += "#define SOFIE_ALPAKA_DIM1D_DEFINED\n";
    fGC += "using Dim1D = alpaka::DimInt<1>;\n";
    fGC += "#endif // SOFIE_ALPAKA_DIM1D_DEFINED\n";
    fGC += "\n#ifndef SOFIE_ALPAKA_WORKDIV_DEFINED\n";
    fGC += "#define SOFIE_ALPAKA_WORKDIV_DEFINED\n";
    fGC += "template<typename TDim, typename TIdx>\n";
    fGC += "inline alpaka::WorkDivMembers<TDim, TIdx> sofie_workdiv(\n";
    fGC += "    alpaka::Vec<TDim, TIdx> const& numElems, TIdx blockSz = TIdx{256})\n{\n";
    fGC += "    auto const numBlocks = alpaka::Vec<TDim, TIdx>::all(\n";
    fGC += "        (numElems[0] + blockSz - TIdx{1}) / blockSz);\n";
    fGC += "    return alpaka::WorkDivMembers<TDim, TIdx>(\n";
    fGC += "        numBlocks,\n";
    fGC += "        alpaka::Vec<TDim, TIdx>::all(blockSz),\n";
    fGC += "        alpaka::Vec<TDim, TIdx>::all(TIdx{1}));\n";
    fGC += "}\n";
    fGC += "#endif // SOFIE_ALPAKA_WORKDIV_DEFINED\n";

    // Macros that redirect alpaka two-argument device-math calls to direct C/C++ calls
    // when compiled by NVCC's device pass (__CUDA_ARCH__ defined).  The normal alpaka
    // form is used in host compilation and on non-CUDA backends.
    // This prevents the deep template instantiation of alpaka's device-math dispatch
    // from corrupting NVCC's namespace tracking (which would misattribute Session to
    // <unnamed> when multiple model headers are included in the same translation unit).
    fGC += "\n#ifndef SOFIE_ALPAKA_DEVICE_MATH_DEFINED\n";
    fGC += "#define SOFIE_ALPAKA_DEVICE_MATH_DEFINED\n";
    fGC += "#if defined(__CUDA_ARCH__)\n";
    for (const char* fn : {"exp","sqrt","log","sin","cos","abs","pow","atan2","atan",
                           "floor","ceil","round","cbrt","tanh","sinh","cosh"}) {
        std::string sfn = fn;
        // pow and atan2 take two numeric args plus acc; handle with variadic macro.
        // All others: SOFIE_DEVICE_fn(acc, x) -> ::fn(x)
        fGC += "#define SOFIE_DEVICE_" + sfn + "(acc, ...) ::" + sfn + "(__VA_ARGS__)\n";
    }
    fGC += "#else\n";
    for (const char* fn : {"exp","sqrt","log","sin","cos","abs","pow","atan2","atan",
                           "floor","ceil","round","cbrt","tanh","sinh","cosh"}) {
        std::string sfn = fn;
        fGC += "#define SOFIE_DEVICE_" + sfn + "(acc, ...) " + sfn + "(acc, __VA_ARGS__)\n";
    }
    fGC += "#endif // __CUDA_ARCH__\n";
    fGC += "#endif // SOFIE_ALPAKA_DEVICE_MATH_DEFINED\n";

    fGC += "\nnamespace SOFIE_" + fName + "{\n";
}

void RModel_Base::OutputGenerated(std::string filename, bool append) {
    // the model can be appended only if a file name is provided
    if (filename.empty()) {
        // if a file is pr
        filename = fName + ".hxx";
        append = false;
    }
    std::ofstream f;
    if (append)
        f.open(filename, std::ios_base::app);
    else
        f.open(filename);
    if (!f.is_open()) {
        throw std::runtime_error("sofie failed to open file for output generated inference code");
    }
    f << fGC;
    f.close();
}

}//SOFIE
