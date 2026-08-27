#ifndef SOFIE_RMODEL
#define SOFIE_RMODEL

#include <optional>

#include "SOFIE/RModel_Base.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"


namespace SOFIE {

class RModel final : public RModel_Base {

   friend class RModelProfiler;
   friend class RModelProfilerGPU;

private:
   bool fIsInitialized = false;
   bool fIsSubGraph = false;
   bool fUseVDT = false;
   bool fProfile = false;
   bool fLowRankFactorize = false;     // enable low rank factorization of eligible weight matrices (disabled by default)
   float fLowRankRatio = 0.5f;         // rank of the factorization = ratio * min(rows, cols) of the weight matrix
   int fVerbose = 0;
   int fBatchSize = -1;
   long fReadPos = 0;  // reading file position
   size_t fConstantTensorSize = 0; // size  (in Bytes) of the allocated constant tensors
   size_t fWeightsTensorSize = 0;  // size  (in Bytes) of the allocated weight tensors
   size_t fOtherTensorSize = 0;    // size  (in Bytes) of intermediate tensors which are not managed by the memory pool

   // Fragmentation tracking
   size_t fPeakAllocatedGPU = 0;
   size_t fPeakLargestFreeBlockGPU = 0;
   size_t fPeakTotalFreeMemoryGPU = 0;
   double fPeakFragmentationGPU = 0.0;

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
   MemoryPoolInfoGPU fIntermediateMemoryInfoGPU; // For GPU 
   std::unordered_map<std::string_view, size_t> fIntermediateTensorFrequencyLookup;    ///<!  lookup table for intermediate tensor frequency (transient)

   std::string fExtraCodeForDimShapes; // extra code needed for initialization of dynamic parameters (e.g. number of non zero elements in NonZero operator)

   enum class EFusionInputAccess {
      Elementwise,
      Scalar,
      Broadcast
   };
   
   struct FusionExternalInput {
      std::string tensorName;
      EFusionInputAccess access = EFusionInputAccess::Elementwise;
      std::vector<size_t> alignedStrides;
      std::string customIndexExpression;
   };

   struct FusionExecutionSchedule {
      size_t blocksPerGrid = 0;
      size_t threadsPerBlock = 0;
      size_t sharedMemoryBytes = 0;
      size_t maxElementsPerThread = 0;
      size_t treeReductionStages = 0;
      size_t synchronizationPoints = 0;
   };

   // GPU ALPAKA elementwise kernel fusion state (transient, computed in GenerateGPU_ALPAKA)
   struct EltwiseFusionGroup {
      std::vector<size_t> opIndices; ///< dependency-chain operator indices forming this group
      std::vector<FusionExternalInput> externalInputs; ///< tensors entering the fusion group from outside
      std::string outputTensor; ///< tensor defining the fused iteration domain
      std::vector<std::string> outputTensors; ///< tensors materialized by the fused kernel
      std::vector<std::string> internalTensors; ///< tensors kept only as local fused values
      std::vector<FusionExecutionSchedule> executionSchedules; ///< available cooperative execution schedules
      size_t numElements = 0;
      size_t launchOpIndex = 0;
      bool usesIndexedEvaluation = false;

      bool isFused() const { return opIndices.size() > 1; }
      std::string suffix() const {
         std::string s;
         for (auto i : opIndices) s += "_" + std::to_string(i);
         return s;
      }
   };

   struct KernelFusionGroup {
      std::vector<size_t> unitIndices;
      std::vector<EltwiseFusionGroup> branches;
      size_t numElements = 0;
      size_t launchOpIndex = 0;

      bool isFused() const { return branches.size() > 1; }

      std::string suffix() const {
         std::string s;
         for (const auto &branch : branches)
            for (const auto opIdx : branch.opIndices)
               s += "_" + std::to_string(opIdx);
         return s;
      }
   };

   struct FusionTensorUseGraph {
      std::unordered_map<std::string, std::vector<size_t>> consumers;
      std::unordered_map<std::string, size_t> producers;
   };

   struct FusionBuildState {
      EltwiseFusionGroup group;
      std::vector<std::string> producedTensors;
      std::vector<size_t> groupOutputShape;
      std::vector<size_t> currentLogicalShape;
      size_t currentOpIdx = 0;
      bool hasReorganize = false;
   };

   struct FusionStructuralScore {
      size_t launchesRemoved = 0;
      size_t liveRangeExtensionByteSteps = 0;
      size_t eliminatedBytes = 0;
      size_t materializedOutputs = 0;
      size_t externalInputs = 0;
   };

   struct FusionCandidate {
      std::vector<size_t> opIndices;
      std::vector<std::string> externalInputs;
      std::vector<std::string> materializedOutputs;
      std::vector<std::string> internalTensors;
      std::vector<FusionExecutionSchedule> executionSchedules;
      FusionStructuralScore score;
      size_t launchOpIndex = 0;
      std::optional<EltwiseFusionGroup> prebuiltGroup;

      bool isFused() const { return opIndices.size() > 1; }
   };

   struct FusionPlan {
      std::vector<size_t> candidateIndices;
      FusionStructuralScore score;
   };

   std::vector<EltwiseFusionGroup> fEltwiseFusionGroups; ///<!
   std::vector<KernelFusionGroup> fKernelFusionGroups; ///< horizontally fused kernel groups
   std::unordered_map<size_t, size_t> fOpToFusionGroupIdx; ///<!  op_idx -> fusion group index
   std::unordered_map<size_t, size_t> fOpToKernelFusionGroupIdx; ///<! op_idx -> horizontal kernel fusion group index
   std::set<std::string> fFusionIntermediateTensors;        ///<!  intermediate tensors whose alloc is skipped
   std::set<size_t>      fSkipOperators;                    ///<!  ops swallowed by a preceding fusion (e.g. GEMM+LeakyReLU)

   FusionTensorUseGraph BuildFusionTensorUseGraph() const;

   FusionCandidate BuildFusionCandidate(const std::vector<size_t> &opIndices, const FusionTensorUseGraph &tensorUses) const;

   bool IsValidFusionCandidate(const FusionCandidate &candidate, const FusionTensorUseGraph &tensorUses) const;

   std::vector<size_t> EnumerateFusionLaunchIndices(const FusionCandidate &candidate, const FusionTensorUseGraph &tensorUses) const;

   std::vector<FusionCandidate> EnumerateSpecialFusionCandidates(const FusionTensorUseGraph &tensorUses) const;
   
   std::vector<FusionCandidate> EnumerateFusionCandidates(const FusionTensorUseGraph &tensorUses) const;

   std::vector<FusionCandidate> EnumerateLinearFusionCandidates(const FusionTensorUseGraph &tensorUses) const;

   FusionStructuralScore ComputeFusionStructuralScore(const FusionCandidate &candidate, const FusionTensorUseGraph &tensorUses) const;

   std::vector<FusionExecutionSchedule> ComputeFusionExecutionSchedules(const FusionCandidate &candidate) const;

   static size_t ComputeDefaultFusionReductionBlockSize(size_t reducedLength);

   size_t ComputeFusionLiveRangeExtensionByteSteps(const FusionCandidate &candidate, const FusionTensorUseGraph &tensorUses) const;

   size_t ComputeFusionPlanLiveRangeExtensionByteSteps(const std::vector<size_t> &candidateIndices, const std::vector<FusionCandidate> &candidates) const;

   bool FusionCandidatesConflict(const FusionCandidate &left, const FusionCandidate &right, const FusionTensorUseGraph &tensorUses) const;

   FusionPlan SelectFusionPlan(const std::vector<FusionCandidate> &candidates, const FusionTensorUseGraph &tensorUses) const;

   static bool IsBetterFusionStructuralScore(const FusionStructuralScore &left, const FusionStructuralScore &right);

   bool IsFuseSafeIntermediate(const std::string &tensorName, const FusionTensorUseGraph &tensorUses) const;

   FusionBuildState InitializeFusionBuildState(size_t firstOpIdx) const;

   bool TryExtendFusionBuildState(FusionBuildState &state, const FusionTensorUseGraph &tensorUses,
                              const std::vector<bool> *blockedOps, bool allowReorganize = true) const;

   EltwiseFusionGroup BuildEltwiseFusionGroup(const FusionCandidate &candidate) const;

   std::vector<EltwiseFusionGroup> BuildKernelFusionLaunchUnits(const FusionTensorUseGraph &tensorUses) const;
   bool CanHorizontallyFuse(const std::vector<EltwiseFusionGroup> &branches, const FusionTensorUseGraph &tensorUses,
                         size_t &launchOpIndex) const;

   bool GetKernelFusionLaunchWindow(const EltwiseFusionGroup &unit, const FusionTensorUseGraph &tensorUses,
                                 size_t &earliestLaunchOpIndex, size_t &latestLaunchOpIndex) const;

   std::vector<KernelFusionGroup> EnumerateKernelFusionGroups(const std::vector<EltwiseFusionGroup> &units,
                                                           const FusionTensorUseGraph &tensorUses) const;

   void ComputeEltwiseFusionGroups();

   size_t ComputeKernelFusionLiveRangeExtensionByteSteps(const KernelFusionGroup &candidate) const;

   std::vector<KernelFusionGroup> SelectKernelFusionGroups(std::vector<KernelFusionGroup> candidates) const;

   std::string GenerateFusedEltwiseLaunch_GPU_ALPAKA(const EltwiseFusionGroup &group) const;

   std::string GenerateFusedEltwiseKernel_GPU_ALPAKA(const EltwiseFusionGroup &group) const;

   std::string GenerateFusedReductionLaunch_GPU_ALPAKA(const EltwiseFusionGroup &group, size_t reductionOpIdx) const;

   std::string GenerateFusedReductionKernel_GPU_ALPAKA(const EltwiseFusionGroup &group, size_t reductionOpIdx) const;

   std::string GenerateFusionValueAtIndex(const EltwiseFusionGroup &group, const std::string &tensorName,
      const std::string &logicalIndex, const std::unordered_map<std::string, size_t> &groupProducers,
      const std::unordered_map<std::string, size_t> &externalInputIndices, std::unordered_map<std::string, std::string> &valueCache,
      std::string &kernelCode, size_t &valueCounter, const std::unordered_map<std::string, std::string> *valueOverrides = nullptr) const;
   std::string GenerateFusionInputIndex(const std::string &inputName, const std::vector<size_t> &outputShape, const std::string &outputIndex) const;

   std::string GenerateKernelFusionLaunch_GPU_ALPAKA(const KernelFusionGroup &group) const;
   std::string GenerateKernelFusionKernel_GPU_ALPAKA(const KernelFusionGroup &group) const;

   bool ResolveFusionInputAccess(const std::string &tensorName, const std::vector<size_t> &outputShape,
                              EFusionInputAccess &access, std::vector<size_t> &alignedStrides) const;

   bool IsSupportedFusionOperator(size_t opIdx, bool allowShuffle, bool allowReorganize, bool allowManyToMany = false) const;

   void AddFusionExternalInput(EltwiseFusionGroup &group, const FusionExternalInput &input) const;

   void InitializeFusionGroup(size_t firstOpIdx, EltwiseFusionGroup &group,
                              std::vector<std::string> &producedTensors, std::vector<size_t> &groupOutputShape) const;
   /// GPU-only pass: fuse GEMM→LeakyReLU (and GEMM→ReLU where not already
   /// handled by the ONNX parser) into a single in-place kernel sequence.
   void FuseGemmActivations_GPU();

   void UpdatePeakAllocatorStats();

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
   std::string ResolveAliasTensor(const std::string &tensorName) const;

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
   // Check if a tensor is a weight tensor (initialized, not constant, and writable)
   bool IsWeightTensor(const std::string &name) const;
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


   // GPU memory allocation
   std::string AllocateIntermediateMemory_GPU_ALPAKA(std::span<const std::string> op_output_tensors, bool keepFusionIntermediates = false);

   void CheckAndFlushIntermediateMemory_GPU_ALPAKA(std::span<const std::string> op_input_tensors,
                                                   const size_t& op_idx);

   void GenerateIntermediateMemoryPool_GPU_ALPAKA();

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
   void GeneratePersistentTensorInfo_GPU_ALPAKA();

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

   // Low rank factorization of eligible weight matrices (Gemm/MatMul). Disabled by default;
   // enable by passing Options::kLowRankFactorize to Generate(). The rank used for each
   // eligible weight matrix is fLowRankRatio * min(rows, cols), rounded down (min 1).
   bool LowRankFactorize() const { return fLowRankFactorize; }
   void SetLowRankRatio(float ratio) { fLowRankRatio = ratio; }
   float LowRankRatio() const { return fLowRankRatio; }

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
