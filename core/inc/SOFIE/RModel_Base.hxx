#ifndef SOFIE_RMODEL_BASE
#define SOFIE_RMODEL_BASE

#include <type_traits>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <memory>
#include <ctime>
#include <set>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "SOFIE/SOFIE_common.hxx"

#ifdef SOFIE_SUPPORT_ROOT_BINARY
#include "TBuffer.h"
#endif


namespace SOFIE {

enum class Options {
   kDefault = 0x0,
   kNoSession = 0x1,
   kNoWeightFile = 0x2,
   kRootBinaryWeightFile = 0x4,
   kGNN = 0x8,
   kGNNComponent = 0x10,
   kProfile = 0x20,
   kBinaryWeightFile = 0x40,
   kTextWeightFile = 0x80,
};

// Optimization levels inspired by ONNXRuntime.
// We only get Operator Fusion with the Basic, and
// memory reuse with Extended. kExtended is enabled
// by default
enum class OptimizationLevel {
   kBasic = 0x0,
   kExtended = 0x1,
};

enum class WeightFileType { None, RootBinary, Text, Binary };


inline std::underlying_type_t<Options> operator|(Options opA, Options opB) {
    return static_cast<std::underlying_type_t<Options>>(opA) |
           static_cast<std::underlying_type_t<Options>>(opB);
}

inline std::underlying_type_t<Options> operator|(std::underlying_type_t<Options> opA, Options opB) {
    return opA | static_cast<std::underlying_type_t<Options>>(opB);
}

class RModel_Base {

protected:
   std::string fFileName;  // file name of original model file for identification
   std::string fParseTime; // UTC date and time string at parsing

   WeightFileType fWeightFile = WeightFileType::Text;

   std::unordered_set<std::string> fNeededBlasRoutines;

   const std::unordered_set<std::string> fAllowedStdLib = {"vector", "algorithm", "cmath", "memory", "span"};
   std::unordered_set<std::string> fNeededStdLib = {"vector"};
   std::unordered_set<std::string> fCustomOpHeaders;

   std::string fName = "UnnamedModel";
   std::string fGC; // generated code
   bool fUseWeightFile = true;
   bool fUseSession = true;
   bool fIsGNN = false;
   bool fIsGNNComponent = false;

   // Function to generate the code for declaring and initializing constant tensors
   // This is for tensors which are not part of weight files and can be created from the Constant operator
   template <typename T>
   std::string GenerateConstantTensorCode(const std::pair<std::string, InitializedTensor> &t)
   {
      std::stringstream strs;
      std::string type = ConvertTypeToString(t.second.type());
      size_t length = ConvertShapeToLength(t.second.shape());
      std::cout<<"Constant tensor name: "<<t.first<<", Constant tensor length: "<<length<<"\n";
      // avoid using stack sizes for constant tensors to reduce compilation time
      bool allocateOnStack = (length > 100) ? false : true;

      const T *data = t.second.data<T>();

      // and check if all values are the same
      bool sameData = false;
      // for non stack allocation check if data are the same
      if (!allocateOnStack && length > 1) {
         size_t idx = 1;
         std::cout<<"insider allocate on stack and length\n";
         do {
            sameData = (data[idx] == data[idx - 1]);
            idx++;
         } while (sameData && idx < length);
      }
      if (allocateOnStack) {
         strs << type << " tensor_" << t.first << "[" << length << "] = " << ConvertValuesToString(length, data) << ";\n";
      } else {
         strs << "std::vector<" << type << "> fTensor_" << t.first << " = ";
         if (sameData)
            strs << "std::vector<" << type << ">(" << length << ", " << ConvertValToString(data[0]) << ");\n";
         else {
            strs << ConvertValuesToString(length, data) << ";\n";
         }
         strs << "const " << type << " * tensor_" + t.first + " = fTensor_" + t.first + ".data();\n";
      }
      return strs.str();
   }

public:
   /**
       Default constructor. Needed to allow serialization of ROOT objects. See
       https://root.cern/manual/io_custom_classes/#restrictions-on-types-root-io-can-handle
   */
   RModel_Base() = default;

   RModel_Base(std::string name, std::string parsedtime);

   // For GNN Functions usage
   RModel_Base(std::string function_name) : fName(function_name) {}

   void AddBlasRoutines(std::vector<std::string> routines)
   {
      for (auto &routine : routines) {
         fNeededBlasRoutines.insert(routine);
      }
   }
   void AddNeededStdLib(std::string libname)
   {
      // if the library is already in the set, insert does nothing
      fNeededStdLib.insert(libname);
   }
   void AddNeededCustomHeader(std::string filename)
   {
       fCustomOpHeaders.insert(filename);
   }
   void GenerateHeaderInfo(std::string &hgname);
   void GenerateHeaderInfo_GPU_ALPAKA(std::string& hgname);
   void PrintGenerated() { std::cout << fGC; }

   std::string ReturnGenerated() { return fGC; }
   void OutputGenerated(std::string filename = "", bool append = false);
   void SetFilename(std::string filename) { fName = filename; }
   std::string GetFilename() { return fName; }
   const std::string & GetName() const { return fName;}
};

enum class GraphType { INVALID = 0, GNN = 1, GraphIndependent = 2 };

enum class FunctionType { UPDATE = 0, AGGREGATE = 1 };
enum class FunctionTarget { INVALID = 0, NODES = 1, EDGES = 2, GLOBALS = 3 };
enum class FunctionReducer { INVALID = 0, SUM = 1, MEAN = 2 };
enum class FunctionRelation { INVALID = 0, NODES_EDGES = 1, NODES_GLOBALS = 2, EDGES_GLOBALS = 3 };

class RModel_GNNBase : public RModel_Base {
public:
   /**
       Default constructor. Needed to allow serialization of ROOT objects. See
       https://root.cern/manual/io_custom_classes/#restrictions-on-types-root-io-can-handle
   */
   RModel_GNNBase() = default;
   virtual void Generate() = 0;
   virtual ~RModel_GNNBase() = default;
};

} // namespace SOFIE

#endif // SOFIE_RMODEL_BASE
