#ifndef SOFIE_RMODELEXTENSION
#define SOFIE_RMODELEXTENSION

namespace SOFIE {

// State a pass library attaches to a model. The model owns the lifetime and never
// inspects the contents, so the pass library's types stay out of the model's header.
class RModelExtension {
public:
   virtual ~RModelExtension() = default;
};

} // namespace SOFIE

#endif
