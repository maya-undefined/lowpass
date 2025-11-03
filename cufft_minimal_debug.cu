#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>  // cufftGetProperty (optional)
#include <cstring>

static const char* cufftErrStr(cufftResult r) {
  switch (r) {
    case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
#if defined(CUFFT_INVALID_DEVICE)
    case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
#endif
    case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
    default: return "CUFFT_UNKNOWN_ERROR";
  }
}

#define CUDA_CHECK(x) do { auto _e = (x); if (_e != cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1);} } while(0)

#define CUFFT_CHECK_LABELED(label, x) do { cufftResult _r=(x); if (_r!=CUFFT_SUCCESS){ \
  fprintf(stderr,"cuFFT error at %s: %s (%d)\n", label, cufftErrStr(_r), (int)_r); exit(1);} } while(0)

// since our sample rate is 65 kHz, and our k_cut is .15 * N/2,
// then this will lowpass remove out everything over 4.8 kHz
__global__ void lowpass_inplace(cufftComplex* X, int nfreq, int k_cut) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < nfreq && k > k_cut) X[k] = make_cuFloatComplex(0.f,0.f);
}

// Scale by 1/N after C2R
// cuda uses un-noralized, which has
// C2R( R2C(x) ) -> N*x, so we have to renormalize
__global__ void scale_real(float* x, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) x[i] *= 1.0f / float(N);
}

static void usage(const char* prog){
  fprintf(stderr,
    "Usage: %s [--sr KHZ] [--fc HZ] [--alpha FRACTION] [--N SAMPLES]\n"
    "  --sr KHZ       Sampling rate in kHz (default 48 => 48000 Hz)\n"
    "  --fc HZ        Cutoff frequency in Hz (overrides --alpha if both set)\n"
    "  --alpha A      Fraction of Nyquist in (0,1); default 0.15 (keep 15%% of 0..Nyquist)\n"
    "  --N SAMPLES    FFT length (default 65536)\n"
    , prog);
}

static bool parse_arg(int& i, int argc, char** argv,
                      const char* flag, double& out){
  if (std::strcmp(argv[i], flag)==0 && i+1<argc){
    out = std::atof(argv[++i]); return true;
  }
  return false;
}

static bool parse_arg_int(int& i, int argc, char** argv,
                          const char* flag, int& out){
  if (std::strcmp(argv[i], flag)==0 && i+1<argc){
    out = std::atoi(argv[++i]); return true;
  }
  return false;
}


int main(int argc, char **argv) {

  int N = 1 << 16;            // amplitude vs time
  const float f_sig = 1000.f;
  const float noise_std = 0.3f;

  double sr_khz = 48.0;     // --sr 48  => 48000 Hz
  double fc_hz  = -1.0;     // cutoff in Hz (if set)
  double alpha  = 0.15;     // fraction of Nyquist (if fc not set)

  // Parse args
  for (int i=1; i<argc; ++i){
    if (std::strcmp(argv[i], "--help")==0 || std::strcmp(argv[i], "-h")==0){ usage(argv[0]); return 0; }
    else if (parse_arg(i, argc, argv, "--sr", sr_khz)) {}
    else if (parse_arg(i, argc, argv, "--fc", fc_hz)) {}
    else if (parse_arg(i, argc, argv, "--alpha", alpha)) {}
    else if (parse_arg_int(i, argc, argv, "--N", N)) {}
    else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
  }
  if (N <= 0) { fprintf(stderr,"N must be positive\n"); return 1; }
  if (alpha <= 0.0 || alpha >= 1.0) {
    // Allow alpha default if fc is set; otherwise clamp
    if (fc_hz < 0.0) { fprintf(stderr,"alpha must be in (0,1). Try --alpha 0.15\n"); return 1; }
  }


  // debugging bullshit since my 4090 has AMP which is too new for ubuntu 22's default cuda-toolkit
  int devCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&devCount));
  if (devCount == 0) { fprintf(stderr,"No CUDA devices found.\n"); return 1; }
  CUDA_CHECK(cudaSetDevice(0));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d), globalMem=%.1f GB\n",
         prop.name, prop.major, prop.minor, prop.totalGlobalMem/1e9);

  int cufftVer = 0, cudaRtVer = 0, cudaDrvVer = 0;
  cufftGetVersion(&cufftVer);
  cudaRuntimeGetVersion(&cudaRtVer);
  cudaDriverGetVersion(&cudaDrvVer);
  printf("cuFFT version: %d, CUDA runtime: %d, CUDA driver: %d\n",
         cufftVer, cudaRtVer, cudaDrvVer);

  const double sr = sr_khz * 1000.0;     // Hz
  const int nfreq = N/2 + 1;

  // Choose k_cut
  int k_cut = 0;
  double fc_used_hz = 0.0;
  if (fc_hz >= 0.0) {
    // fc in Hz → bin
    double kf = std::floor(fc_hz * N / sr);
    if (kf < 0.0) kf = 0.0;
    if (kf > N/2) kf = N/2;
    k_cut = static_cast<int>(kf);
    fc_used_hz = k_cut * (sr / double(N));
    if (alpha > 0.0 && alpha < 1.0) {
      fprintf(stderr, "[note] --fc provided; overriding --alpha\n");
    }
  } else {
    // alpha of Nyquist → bin
    double kf = std::floor(alpha * (N/2));
    if (kf < 0.0) kf = 0.0;
    if (kf > N/2) kf = N/2;
    k_cut = static_cast<int>(kf);
    fc_used_hz = k_cut * (sr / double(N));
  }

  printf("Config:\n");
  printf("  N        = %d samples\n", N);
  printf("  sr       = %.3f kHz (%.0f Hz)\n", sr_khz, sr);
  printf("  k_cut    = %d (out of %d bins)\n", k_cut, nfreq-1);
  printf("  f_cut    = %.3f Hz (Δf = sr/N = %.6f Hz)\n", fc_used_hz, sr/double(N));


  // make up a sine wave data with guass
  std::vector<float> h_in(N), h_out(N);
  std::mt19937 rng(123);
  std::normal_distribution<float> gauss(0.f, noise_std);
  for (int i = 0; i < N; ++i) {
    float t = i / sr;
    h_in[i] = sinf(2.f * M_PI * f_sig * t) + gauss(rng);
  }

  // this will hold the h_in in th ekernels
  float* d_time = nullptr;
  // the fft-ized data in complex form
  cufftComplex* d_freq = nullptr;
  CUDA_CHECK(cudaMalloc(&d_time, sizeof(float) * N));
  CUDA_CHECK(cudaMalloc(&d_freq, sizeof(cufftComplex) * nfreq));
  CUDA_CHECK(cudaMemcpy(d_time, h_in.data(), sizeof(float)*N, cudaMemcpyHostToDevice));

  // create a blank plan handle
  cufftHandle pfwd;
  CUFFT_CHECK_LABELED("cufftCreate(pfwd)", cufftCreate(&pfwd));

  size_t workSizeF = 0;
  // make a 1D R2C plan with single batch (idist/odist implied for 1 batch)
  // define plan: 1d, length n, real->complex, batch=1
  CUFFT_CHECK_LABELED("cufftMakePlan1d(R2C)", cufftMakePlan1d(pfwd, N, CUFFT_R2C, 1, &workSizeF));
  printf("Forward plan work area: %zu bytes\n", workSizeF);

  // inverse path, same plan; go back to real numbers from complex
  cufftHandle pinv;
  CUFFT_CHECK_LABELED("cufftCreate(pinv)", cufftCreate(&pinv));

  size_t workSizeI = 0;
  // inverse plan, length N, complex->real
  CUFFT_CHECK_LABELED("cufftMakePlan1d(C2R)", cufftMakePlan1d(pinv, N, CUFFT_C2R, 1, &workSizeI));
  printf("Inverse plan work area: %zu bytes\n", workSizeI);

  // run the forward plan
  CUFFT_CHECK_LABELED("cufftExecR2C", cufftExecR2C(pfwd, (cufftReal*)d_time, (cufftComplex*)d_freq));

  // actually filter our data
  int tpb = 256;
  int b = (nfreq + tpb - 1) / tpb;
  lowpass_inplace<<<b, tpb>>>(d_freq, nfreq, k_cut);
  CUDA_CHECK(cudaPeekAtLastError());

  // execute reverse plan
  CUFFT_CHECK_LABELED("cufftExecC2R", cufftExecC2R(pinv, (cufftComplex*)d_freq, (cufftReal*)d_time));

  // we got results but we need to renormalize it
  int b2 = (N + tpb - 1) / tpb;
  scale_real<<<b2, tpb>>>(d_time, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_time, sizeof(float)*N, cudaMemcpyDeviceToHost));

  // display a few samples
  for (int i = 0; i < 10; ++i)
    printf("%2d: %8.4f -> %8.4f\n", i, h_in[i], h_out[i]);

  cufftDestroy(pfwd);
  cufftDestroy(pinv);
  cudaFree(d_time);
  cudaFree(d_freq);
  return 0;
}
