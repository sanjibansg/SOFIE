#ifndef SOFIE_RFUNCTION_MEAN
#define SOFIE_RFUNCTION_MEAN

#include "SOFIE/RFunction.hxx"

namespace SOFIE {

class RFunction_Mean: public RFunction_Aggregate {

public:
    RFunction_Mean():RFunction_Aggregate(FunctionReducer::MEAN) {
        fFuncName = "Aggregate_by_Mean";
    }

    std::string GenerateModel();
};

} //SOFIE

#endif //SOFIE_RFUNCTION_MEAN
