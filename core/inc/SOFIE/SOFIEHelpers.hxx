#ifndef SOFIE_SOFIE_HELPERS
#define SOFIE_SOFIE_HELPERS


#include <type_traits>
#include <utility>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>



///Helper class used by SOFIEFunctor to wrap the
///infer signature interface to RDataFrame
template <typename I, typename F, typename T>
class SofieFunctorHelper;

template <std::size_t... N,  typename Session_t, typename T>
class SofieFunctorHelper<std::index_sequence<N...>, Session_t, T> {
   /// this is the magic to define the operator() with N fixed parameter arguments
   template <std::size_t Idx>
   using AlwaysT = T;

   std::vector<std::vector<T>> fInput;
   std::vector<Session_t> fSessions;

public:

   SofieFunctorHelper(unsigned int nslots = 0, const std::string & filename = "") :
      fInput(1)
   {
      // create Sessions according to given number of slots.
      // if number of slots is zero create a single session
      if (nslots < 1) nslots = 1;
      fInput.resize(nslots);
      fSessions.reserve(nslots);
      for (unsigned int i = 0; i < nslots; i++) {
         if (filename.empty())
            fSessions.emplace_back();
         else
            fSessions.emplace_back(filename);
      }
   }

   double operator()(unsigned slot, AlwaysT<N>... args) {
      fInput[slot] = {args...};
      auto y =  fSessions[slot].infer(fInput[slot].data());
      return y[0];
   }
};

/// SofieFunctor : used to wrap the infer function of the
/// generated model by SOFIE in a RDF compatible signature.
/// The number of slots is an optional parameter used to
/// create multiple SOFIE Sessions, which can be run in a parallel
/// model evaluation. One should use as number of slots the number of slots used by
/// RDataFrame. By default, in case of `nslots=0`, only a single Session will be created
/// and the Functor cannot be run in parallel.
/// Examples of using the SofieFunctor are the C++ tutorial SOFIE_RDataFrame.C
/// and the Python tutorial SOFIE_RDataFrame.py which makes use of the ROOT JIT
/// to compile on the fly the generated SOFIE model.
template <std::size_t N, typename Session_t>
auto SofieFunctor(unsigned int nslots = 0, const std::string & weightsFile = "") -> SofieFunctorHelper<std::make_index_sequence<N>, Session_t, float>
{
   return SofieFunctorHelper<std::make_index_sequence<N>, Session_t, float>(nslots, weightsFile);
}

namespace SOFIE {

namespace Internal {

// Cyclic Jacobi eigenvalue algorithm for a small dense symmetric matrix S (n x n, row-major).
// On return, S has been overwritten with the eigenvalues on its diagonal, and the columns
// of V (n x n, row-major, i.e. V[i*n+j] is component i of eigenvector j) hold the (already
// orthonormal) eigenvectors. Intended for small n only (a handful to a few dozen).
inline void JacobiEigenSymmetric(std::vector<double> &S, std::vector<double> &V, std::size_t n)
{
   V.assign(n * n, 0.0);
   for (std::size_t i = 0; i < n; i++)
      V[i * n + i] = 1.0;

   const int maxSweeps = 100;
   for (int sweep = 0; sweep < maxSweeps; sweep++) {
      double off = 0.0;
      for (std::size_t p = 0; p < n; p++)
         for (std::size_t q = p + 1; q < n; q++)
            off += S[p * n + q] * S[p * n + q];
      if (off < 1e-30)
         break;

      for (std::size_t p = 0; p < n; p++) {
         for (std::size_t q = p + 1; q < n; q++) {
            double apq = S[p * n + q];
            if (std::abs(apq) < 1e-300)
               continue;
            double app = S[p * n + p];
            double aqq = S[q * n + q];
            double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
            double c = std::cos(phi);
            double s = std::sin(phi);
            // rotate rows/columns p and q of S
            for (std::size_t k = 0; k < n; k++) {
               double skp = S[k * n + p];
               double skq = S[k * n + q];
               S[k * n + p] = c * skp - s * skq;
               S[k * n + q] = s * skp + c * skq;
            }
            for (std::size_t k = 0; k < n; k++) {
               double spk = S[p * n + k];
               double sqk = S[q * n + k];
               S[p * n + k] = c * spk - s * sqk;
               S[q * n + k] = s * spk + c * sqk;
            }
            // accumulate rotation into eigenvector matrix
            for (std::size_t k = 0; k < n; k++) {
               double vkp = V[k * n + p];
               double vkq = V[k * n + q];
               V[k * n + p] = c * vkp - s * vkq;
               V[k * n + q] = s * vkp + c * vkq;
            }
         }
      }
   }
}

} // namespace Internal

/// Compute a rank-`rank` approximation of a dense row-major matrix W (rows x cols):
///    W  ~=  A * B
/// where A is (rows x rank) and B is (rank x cols), using randomized truncated SVD
/// (Halko-Martinsson-Tropp). A folds in the singular values (A = U_r * Sigma_r), B holds
/// the right singular vectors (B = V_r^T), so a single extra matrix product reconstructs
/// the approximation.
///
/// Returns false (leaving A/B unmodified) if `rank` is not strictly smaller than
/// min(rows, cols), i.e. when no compression is possible.
inline bool ComputeLowRankFactors(const float *W, std::size_t rows, std::size_t cols, std::size_t rank,
                                   std::vector<float> &A, std::vector<float> &B)
{
   std::size_t minDim = std::min(rows, cols);
   if (rank == 0 || rank >= minDim)
      return false;

   // oversample for better accuracy of the randomized range finder
   std::size_t l = std::min(minDim, rank + 8);

   std::mt19937 rng(42); // fixed seed for reproducible codegen
   std::normal_distribution<double> gauss(0.0, 1.0);

   // Omega: cols x l random Gaussian matrix
   std::vector<double> Omega(cols * l);
   for (auto &v : Omega)
      v = gauss(rng);

   // Y = W * Omega : rows x l
   auto matmul = [](const float *M, std::size_t mr, std::size_t mc, const std::vector<double> &X, std::size_t xc,
                     std::vector<double> &out) {
      out.assign(mr * xc, 0.0);
      for (std::size_t i = 0; i < mr; i++)
         for (std::size_t k = 0; k < mc; k++) {
            double mik = M[i * mc + k];
            if (mik == 0.0)
               continue;
            for (std::size_t j = 0; j < xc; j++)
               out[i * xc + j] += mik * X[k * xc + j];
         }
   };
   auto matmulT = [](const float *M, std::size_t mr, std::size_t mc, const std::vector<double> &X, std::size_t xc,
                      std::vector<double> &out) {
      // out = M^T * X  where M is (mr x mc), X is (mr x xc), out is (mc x xc)
      out.assign(mc * xc, 0.0);
      for (std::size_t k = 0; k < mr; k++)
         for (std::size_t i = 0; i < mc; i++) {
            double mki = M[k * mc + i];
            if (mki == 0.0)
               continue;
            for (std::size_t j = 0; j < xc; j++)
               out[i * xc + j] += mki * X[k * xc + j];
         }
   };
   auto orthonormalize = [](std::vector<double> &Q, std::size_t qr, std::size_t qc) {
      // modified Gram-Schmidt on the qc columns of Q (qr x qc, row-major)
      for (std::size_t j = 0; j < qc; j++) {
         for (std::size_t prev = 0; prev < j; prev++) {
            double dot = 0.0;
            for (std::size_t i = 0; i < qr; i++)
               dot += Q[i * qc + prev] * Q[i * qc + j];
            for (std::size_t i = 0; i < qr; i++)
               Q[i * qc + j] -= dot * Q[i * qc + prev];
         }
         double norm = 0.0;
         for (std::size_t i = 0; i < qr; i++)
            norm += Q[i * qc + j] * Q[i * qc + j];
         norm = std::sqrt(norm);
         if (norm < 1e-300)
            norm = 1e-300;
         for (std::size_t i = 0; i < qr; i++)
            Q[i * qc + j] /= norm;
      }
   };

   std::vector<double> Q;
   matmul(W, rows, cols, Omega, l, Q); // Q: rows x l
   orthonormalize(Q, rows, l);

   // a couple of power iterations improve accuracy for slowly-decaying spectra
   std::vector<double> Z;
   for (int iter = 0; iter < 2; iter++) {
      matmulT(W, rows, cols, Q, l, Z); // Z = W^T * Q : cols x l
      orthonormalize(Z, cols, l);
      matmul(W, rows, cols, Z, l, Q); // Q = W * Z : rows x l
      orthonormalize(Q, rows, l);
   }

   // small matrix Bs = Q^T * W : l x cols
   std::vector<double> Bs(l * cols, 0.0);
   for (std::size_t i = 0; i < rows; i++)
      for (std::size_t k = 0; k < l; k++) {
         double qik = Q[i * l + k];
         if (qik == 0.0)
            continue;
         for (std::size_t j = 0; j < cols; j++)
            Bs[k * cols + j] += qik * W[i * cols + j];
      }

   // eigen-decompose the small l x l Gram matrix G = Bs * Bs^T
   std::vector<double> G(l * l, 0.0);
   for (std::size_t i = 0; i < l; i++)
      for (std::size_t j = 0; j < l; j++) {
         double s = 0.0;
         for (std::size_t k = 0; k < cols; k++)
            s += Bs[i * cols + k] * Bs[j * cols + k];
         G[i * l + j] = s;
      }
   std::vector<double> V; // l x l eigenvectors
   Internal::JacobiEigenSymmetric(G, V, l);

   // rank the eigenvalues (diagonal of G after Jacobi) in descending order, keep top `rank`
   std::vector<std::size_t> order(l);
   for (std::size_t i = 0; i < l; i++)
      order[i] = i;
   std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return G[a * l + a] > G[b * l + b]; });

   A.assign(rows * rank, 0.0f);
   B.assign(rank * cols, 0.0f);

   for (std::size_t r = 0; r < rank; r++) {
      std::size_t idx = order[r];
      double eigenval = std::max(0.0, G[idx * l + idx]);
      double sigma = std::sqrt(eigenval);

      // right singular vector (in cols-space): v_r = (1/sigma) * Bs^T * u_idx , u_idx = V[:,idx]
      std::vector<double> vr(cols, 0.0);
      if (sigma > 1e-12) {
         for (std::size_t k = 0; k < l; k++) {
            double uk = V[k * l + idx];
            if (uk == 0.0)
               continue;
            for (std::size_t j = 0; j < cols; j++)
               vr[j] += uk * Bs[k * cols + j];
         }
         for (std::size_t j = 0; j < cols; j++)
            vr[j] /= sigma;
      }
      for (std::size_t j = 0; j < cols; j++)
         B[r * cols + j] = static_cast<float>(vr[j]);

      // left singular vector (in rows-space): u_r = Q * u_idx , folding sigma into A
      for (std::size_t i = 0; i < rows; i++) {
         double ui = 0.0;
         for (std::size_t k = 0; k < l; k++)
            ui += Q[i * l + k] * V[k * l + idx];
         A[i * rank + r] = static_cast<float>(ui * sigma);
      }
   }

   return true;
}

} // namespace SOFIE

#endif //SOFIE_SOFIE_HELPERS
