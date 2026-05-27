#ifndef SOFIE_RMODEL
#define SOFIE_RMODEL

#include "SOFIE/RModel_Base.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"


namespace SOFIE {

class RModel final : public RModel_Base {

private:
   bool fIsInitialized = false;
   bool fIsSubGraph = false;
   bool fUseVDT = false;
   int fVerbose = 0;
   int fBatchSize = -1;
   long fReadPos = 0;  // reading file position
   size_t fConstantTensorSize = 0; // size  (in Bytes) of the allocated constant tensors
   size_t fWeightsTensorSize = 0;  // size  (in Bytes) of the allocated weight tensors
   size_t fOtherTensorSize = 0;    // size  (in Bytes) of intermediate tensors which are not managed by the memory pool

   OptimizationLevel fOptimizationLevel = OptimizationLevel::kExtended;

   std::unordered_map<std::string, InputTensorInfo> fInputTensorInfos; // input tensors where shape may not fully defined or other graph inputs?
   std::unordered_map<std::string, TensorInfo> fReadyInputTensorInfos; // input tensors where shape is full defined
   std::unordered_map<std::string, InitializedTensor> fInitializedTensors;
   std::unordered_map<std::string, TensorInfo> fIntermediateTensorInfos;
   std::unordered_map<std::string, DynamicTensorInfo> fDynamicTensorInfos;
   std::unordered_map<std::string, std::pair<std::vector<Dim>, bool>> fShapeTensors; // constant tensors describing a shape
   std::unordered_map<std::string, std::string> fAliasTensors; // alias tensors (name -> original tensor name)
   std::unordered_map<std::string, std::string>
      fShapeParams; // parameters defining the dynamic shape (e.g. batch size), store also its default value
   std::vector<std::string> fDimShapeNames; // parameter names used to define the shapes
   std::vector<std::string> fOutputTensorNames;
   std::vector<std::string> fInputTensorNames; // input tensor names using ONNX order

   std::vector<std::unique_ptr<ROperator>> fOperators;

   std::vector<std::shared_ptr<RModel>> fSubGraphs;    ///<!  sub-graph models (transient)
   RModel * fParentGraph = nullptr;

   const std::string SP = "   ";

   // memory pool information for intermediate tensors
   MemoryPoolInfo fIntermediateMemoryInfo;    ///<!  intermediate memory info (transient)
   std::unordered_map<std::string_view, size_t> fIntermediateTensorFrequencyLookup;    ///<!  lookup table for intermediate tensor frequency (transient)

   std::string fExtraCodeForDimShapes; // extra code needed for initialization of dynamic parameters (e.g. number of non zero elements in NonZero operator)

   // GPU ALPAKA elementwise kernel fusion state (transient, computed in GenerateGPU_ALPAKA)
   struct EltwiseFusionGroup {
      std::vector<size_t> opIndices; ///< consecutive op indices forming this group
      std::string inputTensor;       ///< input tensor name of the first op
      std::string outputTensor;      ///< output tensor name of the last op
      size_t numElements = 0;
      bool isFused() const { return opIndices.size() > 1; }
      std::string suffix() const {
         std::string s;
         for (auto i : opIndices) s += "_" + std::to_string(i);
         return s;
      }
   };
   std::vector<EltwiseFusionGroup> fEltwiseFusionGroups; ///<!
   std::unordered_map<size_t, size_t> fOpToFusionGroupIdx; ///<!  op_idx -> fusion group index
   std::set<std::string> fFusionIntermediateTensors;        ///<!  intermediate tensors whose alloc is skipped
   std::set<size_t>      fSkipOperators;                    ///<!  ops swallowed by a preceding fusion (e.g. GEMM+LeakyReLU)
   void ComputeEltwiseFusionGroups();
   /// GPU-only pass: fuse GEMM→LeakyReLU (and GEMM→ReLU where not already
   /// handled by the ONNX parser) into a single in-place kernel sequence.
   void FuseGemmActivations_GPU();

public:
   // Rule of five: explicitly define move semantics, disallow copy
   RModel(RModel &&other);
   RModel &operator=(RModel &&other);
   RModel(const RModel &other) = delete;
   RModel &operator=(const RModel &other) = delete;
   ~RModel() = default;

   /**
       Default constructor. Needed to allow serialization of ROOT objects. See
       https://root.cern/manual/io_custom_classes/#restrictions-on-types-root-io-can-handle
   */
   RModel() = default;
   RModel(std::string name, std::string parsedtime) : RModel_Base(name, parsedtime) {}

   // For GNN Functions usage
   RModel(std::string function_name) : RModel_Base(function_name) {}

   int Verbose() const { return fVerbose;}

   std::vector<size_t> GetTensorShape(const std::string & name) const;
   std::vector<Dim> GetDimTensorShape(const std::string & name) const;
   ETensorType GetTensorType(std::string name) const;
   std::vector<Dim> GetDynamicTensorShape(const std::string & name) const ;

   // get the values for the tensor representing a shape
   const std::vector<Dim> & GetShapeTensorValues(const std::string & tensor_name) const;


   bool CheckIfTensorAlreadyExist(std::string tensor_name);
   void AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<Dim> shape);
   void AddInputTensorInfo(std::string input_name, ETensorType type, std::vector<size_t> shape);
   void AddOperator(std::unique_ptr<ROperator> op, int order_execution = -1);
   void AddOperatorReference(ROperator *op, int order_execution = -1)
   {
      std::unique_ptr<ROperator> tmp(op);
      AddOperator(std::move(tmp), order_execution);
   }
   void AddInitializedTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape,
                             std::shared_ptr<void> data);
   void AddConstantTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape,
                             std::shared_ptr<void> data);

   template<class T>
   void AddConstantTensor(const std::string & name, const std::vector<size_t> & shape, const T * data) {
      size_t length = ConvertShapeToLength(shape);
      std::shared_ptr<void> data_ptr(malloc(length * sizeof(T)), free);
      std::memcpy(data_ptr.get(), (void*) data, length * sizeof(T));
      std::cout<<"Length of constant tensor "<<name<<" added: "<<length<<std::endl;
      AddConstantTensor(name, GetTemplatedType<T>(T()), shape, data_ptr);
   }
   // for boolean can be more convenient passing an std::vector
   template<class T>
   void AddConstantTensor(const std::string & name, const std::vector<size_t> & shape, const std::vector<T> & data) {
      size_t length = data.size();
      std::shared_ptr<void> data_ptr(malloc(length * sizeof(T)), free);
      std::copy(data.begin(), data.end(), (T*) data_ptr.get());
      //std::memcpy(data_ptr.get(), (void*) data, length * sizeof(T));
      AddConstantTensor(name, GetTemplatedType<T>(T()), shape, data_ptr);
   }

   template <typename T>
   void AddInitializedTensor(const std::string & tensor_name, const std::vector<std::size_t> & shape, T *raw_data)
   {
      size_t size = ConvertShapeToLength(shape);
      std::shared_ptr<void> data(malloc(size * sizeof(T)), free);
      std::memcpy(data.get(), raw_data, size * sizeof(T));
      AddInitializedTensor(tensor_name,  GetTemplatedType(T()), shape, data);
   }

   void AddShapeTensor(const std::string & name, const std::vector<Dim> & shapeValues, bool scalar = false);
   void AddAliasTensor(const std::string & name, const std::string & origin);
   bool IsAliasTensor(const std::string & tensor_name) const;

   void AddExtraCodeForDimShapes(const std::string & code) { fExtraCodeForDimShapes += code; }

   // add and initialize subgraph to the model
   void InitializeSubGraph(std::shared_ptr<RModel>  graph);

   // set a flag to indicate tensor does not need to be written in a weight file
   // (e.g. shape tensors used as input to define a shape (in Reshape))
   void SetNotWritableInitializedTensor(const std::string & tensor_name);

   // Check if a tensor is initialized
   bool IsInitializedTensor(const std::string &name) const;
   // Check if a tensor is Constant (note a Constant tensor is also initialized)
   bool IsConstantTensor(const std::string &name) const;
   bool IsDynamicTensor(const std::string &name) const;
   // Check if tensor is a input dynamic tensor (without a specified shape, based on Sim structure
   bool IsDimInputTensor(const std::string &name) const;
   // check if tensor is a fully specified input tensor
   bool IsReadyInputTensor(const std::string &name) const;
   /// check if a tensor is a shape tensor
   bool IsShapeTensor(const std::string & name) const;

   // Add intermediate tensor
   void AddIntermediateTensor(std::string tensor_name, ETensorType type, std::vector<Dim> dim_shape);
   void AddIntermediateTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape);
   // Add an intermediate dynamic tensor
   void AddDynamicTensor(std::string tensor_name, ETensorType type, std::vector<Dim> shape);
   void AddShapeParam(const std::string & name, size_t def_value = 0);
   void AddInputTensorName(std::string name);
   void AddOutputTensorNameList(std::vector<std::string> output_tensor_names);
   void
   UpdateOutputTensorList(std::vector<std::string> curr_output_tensor, std::vector<std::string> modify_output_tensor);
   void UpdateInitializedTensor(std::string tensor_name, ETensorType type, std::vector<std::size_t> shape,
                                std::shared_ptr<void> data);
   std::shared_ptr<void> GetInitializedTensorData(std::string tensor_name);
   void RemoveInitializedTensor(std::string tensor_name);
   template<class T>
   std::vector<T> GetTensorData(const std::string & name);

   void Initialize(int batchSize = -1, bool verbose = false);
   void Initialize(const std::map<std::string,size_t> & inputParams, bool verbose = false);

   void Generate(std::underlying_type_t<Options> options, int batchSize = -1, long pos = 0, bool verbose = false);
   void Generate(Options options = Options::kDefault, int batchSize = -1, int pos = 0, bool verbose = false)
   {
      Generate(static_cast<std::underlying_type_t<Options>>(options), batchSize, pos, verbose);
   }
   void GenerateGPU_ALPAKA(std::underlying_type_t<Options> options, int batchSize = -1, bool verbose = false);
   void GenerateGPU_ALPAKA(Options options = Options::kDefault, int batchSize = -1, bool verbose = false)
   {
      GenerateGPU_ALPAKA(static_cast<std::underlying_type_t<Options>>(options), batchSize, verbose);
   }
   // generate the infer function signature. If isdecl= false generate the calling infer function
   // used to infer the sub-graphs
   std::string GenerateInferSignature(bool isdecl = true);

   // generate the infer function signature for inference on ALPAKA. If isdecl= false generate the calling infer function
   // used to infer the sub-graphs
   std::string GenerateInferSignature_GPU_ALPAKA(bool isdecl = true);

   // generate the _infer_impl signature using ViewPlainPtr types instead of Buf types
   std::string GenerateImplSignature_GPU_ALPAKA(bool isdecl = true);

   void RemoveIntermediateTensor(const std::string& tensor_name){
      fIntermediateTensorInfos.erase(tensor_name);
   }

   // calculate total intermediate memory and position intermediate tensor addresses
   std::string AllocateIntermediateMemory(std::span<const std::string> op_output_tensors);
   void CheckAndFlushIntermediateMemory(std::span<const std::string> op_output_tensors, const size_t& op_idx);

protected:
   // internal functions
   // generate code for the initialized tensors
   void GenerateInitializedTensorInfo();

   void GenerateInitializedTensorInfo_GPU_ALPAKA(); 
   // generate code for the intermediate tensors
   void GenerateIntermediateTensorInfo();

   // generate code for the temporary initialized tensors containers
   void GenerateTemporaryInitializedTensorContainers_GPU_ALPAKA();

   // generate code for the dynamic tensors
   void GenerateDynamicTensorInfo();

   void GenerateDynamicTensorInfo_GPU_ALPAKA();
   // generate code for declarations needed by operators
   void GenerateOperatorDeclarations();
   // generate code for inference
   void GenerateOutput();

   void GenerateOutput_GPU_ALPAKA();

   void MoveInitializedTensorsToBuffers_ALPAKA();
   // generate code for initializing memory pool for intermediate tensors
   void GenerateIntermediateMemoryPool();
   // Generate all session code
   void GenerateSessionCode();
   void GenerateSessionCode_GPU_ALPAKA();
   void GenerateGPU_ALPAKA_Buffers();

   void CheckAndFuseOperators();
   bool IsInputTensorShapeParam(std::string const &paramName) const;
   std::vector<std::string> CollectTensorMemberNames(const std::string &input);
   void GenerateRequiredInputTensorInfo();

public:
   const std::vector<std::string> &GetInputTensorNames() const { return fInputTensorNames; }
   const std::vector<std::string> &GetOutputTensorNames() const { return fOutputTensorNames; }
   const std::vector<std::string> & GetDimShapeNames() const { return fDimShapeNames; }

   void ReadInitializedTensorsFromFile(long);
   long WriteInitializedTensorsToFile(std::string filename = "");

   void PrintIntermediateTensors() const;
   void PrintOutputTensors() const;
   void PrintSummary() const;
   void OutputGenerated(std::string filename = "", bool append = false);
   std::vector<std::string> GetOutputTensorNames() { return fOutputTensorNames; }
   void SetFilename(std::string filename) { fName = filename; }

   /*
      template <typename T>
      void AddInitializedTensor(std::string tensor_name, RTensor<T> new_tensor){
         //a view only
         T obj;
         if (fInitializedTensors.find(tensor_name) != fInitializedTensors.end()){
            throw std::runtime_error("sofie: initialized tensor with name " + tensor_name + " already exists \n");
         }
         InitializedTensor new_tensor_ {GetTemplatedType(obj), new_tensor.GetShape() ,
      static_cast<void>(new_tensor.GetData())}; fInitializedTensors[tensor_name] = new_tensor_;
      }
   */

   void PrintRequiredInputTensors() const;
   void PrintInitializedTensors() const;
   void PrintDynamicTensors() const;
   void HeadInitializedTensors(std::string name, int n_print = 50);

   bool UseSession() const { return fUseSession; }
   void SetUseVDT(bool on) {
      fUseVDT = on;
   }
   bool UseVDT() const { return fUseVDT;}
   
#ifdef SOFIE_SUPPORT_ROOT_BINARY
   // Use the ClassDef macro to allow definition of custom streaming
   ClassDefNV(RModel, 3);
#endif

};

template<class T>
inline std::vector<T> RModel::GetTensorData(const std::string & name) {
   if (!IsInitializedTensor(name)) return std::vector<T>{};
   T * data = static_cast<T*>(GetInitializedTensorData(name).get());
   size_t size = ConvertShapeToLength(GetTensorShape(name));
   return std::vector<T>(data, data+size);
}

template<>
inline std::vector<Dim> RModel::GetTensorData<Dim>(const std::string & name) {
   if (!IsShapeTensor(name)) return std::vector<Dim>{};
   return GetShapeTensorValues(name);
}


} // namespace SOFIE

#endif // SOFIE_RMODEL
