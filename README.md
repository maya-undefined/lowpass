# cuFFT Low-Pass Demo (1D Real Signals)

A minimal, self-contained example of using **NVIDIA cuFFT** to:

1) generate a noisy sine wave,
2) run a **real→complex FFT (R2C)**,
3) apply a **low-pass filter** by zeroing high-frequency bins,
4) run a **complex→real inverse FFT (C2R)**, and
5) **normalize** by `1/N` to recover time-domain amplitudes.

---

## What is the FFT? What is cuFFT?

- **FFT (Fast Fourier Transform)** converts a time-domain signal `x[n]` into a frequency-domain spectrum `X[k]`. Each bin `k` represents a sinusoid at frequency `f[k] = k · (sr/N)` with a **complex value** whose magnitude is the amplitude and whose angle is the phase.

- **cuFFT** is NVIDIA’s GPU-accelerated FFT library. It provides highly optimized kernels for 1D/2D/3D FFTs and batched transforms. Using cuFFT, large FFTs (or many FFTs) run far faster than on CPU.

> For real inputs, the FFT output is **complex** and exhibits *conjugate symmetry*. cuFFT’s R2C stores only the **non-negative** frequencies `k=0..N/2`, so the spectrum length is `N/2 + 1` complex numbers.

---

## What this program does

- Creates a 1 kHz sine wave with Gaussian noise (length **N = 65,536** by default; sample rate **sr** is configurable).
- Copies it to the GPU (`d_time`).
- **Forward FFT (R2C)** → `d_freq` of length `N/2+1` complex bins.
- **Low-pass kernel** zeros bins with index `k > k_cut` (i.e., frequencies above `f_c`).
- **Inverse FFT (C2R)** back to time domain in `d_time`.
- **Scale kernel** multiplies every sample by `1/N` (cuFFT’s inverse is unnormalized).
- Copies result back to host and prints a few samples.

### Low-pass cutoff

You can specify the cutoff in **Hz** or as a **fraction of Nyquist**:

- **Fraction of Nyquist** `alpha` (0..1):
  ```
  k_cut = floor(alpha * (N/2))
  f_c   = k_cut * (sr / N)
  ```
- **Absolute Hz** `f_c`:
  ```
  k_cut = floor(f_c * N / sr)  // clamped to [0, N/2]
  ```

If both `--fc` and `--alpha` are provided, `--fc` **wins**.

### Why normalization is needed

cuFFT uses the **unnormalized** convention (no `1/N` in either direction). `C2R(R2C(x)) = N * x`. We therefore scale by `1/N` after the inverse to restore amplitudes.

---

## Build & Run

> **Requires CUDA 12.x or newer** (Ada / sm_89 GPUs like RTX 4080 need 12.x). Make sure your `nvcc` and `libcufft.so` come from the same CUDA install.

```bash
# Compile (adjust SM arch if needed)
nvcc -O2 cufft_minimal.cu -o cufft_minimal -lcufft \
     -gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=compute_89

# Examples
./cufft_minimal                         # defaults: sr=48 kHz, alpha=0.15, N=65536
./cufft_minimal --sr 65 --alpha 0.10    # sr=65 kHz, cutoff = 10% of Nyquist
./cufft_minimal --sr 48 --fc 3600       # sr=48 kHz, absolute cutoff 3600 Hz
./cufft_minimal --N 131072 --alpha 0.2  # larger FFT, 20% of Nyquist
```

### Optional: Debug/diagnostics build

There’s a `cufft_minimal_debug.cu` variant that prints GPU/cuFFT/CUDA versions and labels each cuFFT call. Useful if plan creation fails or libraries are mismatched.

```bash
nvcc -O2 cufft_minimal_debug.cu -o cufft_minimal_debug -lcufft
./cufft_minimal_debug
```

---

## Code Tour (high-level)

```cpp
// Host: make noisy sine into h_in[N]
cudaMalloc(d_time, N*sizeof(float));
cudaMalloc(d_freq, (N/2+1)*sizeof(cufftComplex));
cudaMemcpy(d_time, h_in, ...);

// Plans (1D)
cufftPlan1d(plan_fwd, N, CUFFT_R2C, 1);
cufftPlan1d(plan_inv, N, CUFFT_C2R, 1);

// Forward FFT: time → freq
cufftExecR2C(plan_fwd, d_time, d_freq);

// Low-pass in place: zero X[k] for k > k_cut
lowpass_inplace<<<...>>>(d_freq, nfreq, k_cut);

// Inverse FFT: freq → time (unnormalized)
cufftExecC2R(plan_inv, d_freq, d_time);

// Normalize (1/N)
scale_real<<<...>>>(d_time, N);

// Copy result back, print/save
cudaMemcpy(h_out, d_time, ...);
```

### Kernels

- **Low-pass (frequency domain)**
  ```cpp
  __global__ void lowpass_inplace(cufftComplex* X, int nfreq, int k_cut){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nfreq && k > k_cut) { X[k].x = 0.f; X[k].y = 0.f; }
  }
  ```
- **Scale (time domain)**
  ```cpp
  __global__ void scale_real(float* x, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] *= 1.0f / N;
  }
  ```

---

## How to choose `N` and what it means

- `N` = number of time samples; duration `T = N / sr`.
- Frequency resolution `Δf = sr / N`. Larger `N` ⇒ finer frequency spacing.
- R2C output stores `N/2 + 1` complex bins (0..Nyquist).

### Zero-padding
You can zero‑pad the time signal to a larger `N` before FFT to interpolate the spectrum (more bins), but it **does not** increase true resolution beyond `sr/N_original`.

---

## Interpreting results

- A low-pass filter **does not change pitch**; it removes high-frequency noise/harmonics (timbre). If `f_c` > tone frequency, the tone remains but sounds cleaner/warmer.
- If `f_c` < tone frequency, the tone is attenuated or removed.

---

## Common pitfalls & quick fixes

- **CUFFT_INTERNAL_ERROR at plan creation:** Your CUDA toolkit/cuFFT is too old for your GPU (e.g., Ada requires CUDA 12.x). Install CUDA 12.x and relink.
- **Wrong buffer sizes:** R2C output is `(N/2+1)` complex numbers—not `N`.
- **Forgot normalization:** Always scale by `1/N` after inverse (or equivalently elsewhere).
- **Cutoff math:** Convert Hz↔bin with `k = floor(f_c * N / sr)` and clamp to `[0, N/2]`.

---

## Easy extensions

- **CLI args:** already supported → `--sr`, `--fc`, `--alpha`, `--N`.
- **Windowing:** apply a Hann window pre-FFT to reduce spectral leakage.
- **Notch filter:** zero bins around known interference (e.g., 60 Hz ± bw).
- **Batches:** use `cufftPlanMany` and a 2D low-pass kernel (bin × signal) to process 100s of signals.
- **STFT/spectrogram:** frame the signal (e.g., 2048-sample windows with 75% overlap), run batched R2C, plot magnitude.

---

**License:** MIT (or your choice)

**Tested on:** CUDA 12.x, RTX 4080 Laptop (sm_89)

