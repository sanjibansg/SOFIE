#ifndef SOFIE_RFUNCTION_SUM
#define SOFIE_RFUNCTION_SUM


#include "SOFIE/RFunction.hxx"

namespace SOFIE {

class RFunction_Sum: public RFunction_Aggregate {

public:
    RFunction_Sum():RFunction_Aggregate(FunctionReducer::SUM) {
        fFuncName = "Aggregate_by_Sum";
    }

    std::string GenerateModel();
};

} //SOFIE

#endif //SOFIE_RFUNCTION_SUM
