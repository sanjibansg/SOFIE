namespace ReduceL2_ExpectedOutput{
   // Input [1,2,3] = {5,2,3,5,5,4}, ReduceL2 over axis=1, keepdims=0 → shape [1,3]
   // col0: sqrt(5^2+5^2)=sqrt(50), col1: sqrt(2^2+5^2)=sqrt(29), col2: sqrt(3^2+4^2)=5
   float output[] = {
      7.0710678118654755f,
      5.385164807134504f,
      5.0f
   };
} // namespace ReduceL2_ExpectedOutput
