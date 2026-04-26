(pinn_env) PS D:\Work\PHD-2026\Project\project-2\pinn-error-bounds> python run_complete_v2.py
======================================================================
 A POSTERIORI ERROR BOUNDS FOR PINN SOLUTIONS (CORRECTED v2)
 Interior Residual + BC Lifting Decomposition
======================================================================

██████████████████████████████████████████████████████████████████████
█  BENCHMARK 1: POISSON EQUATION (SMOOTH)
██████████████████████████████████████████████████████████████████████

  === Poisson (smooth) | 4x64 ===
    Arch: [64, 64, 64, 64], tanh, fourier=False, params=12737
    Adam   1000 | L=6.369e-02 | Res=4.607e-02 | BC=1.762e-03
    Adam   2000 | L=1.976e-03 | Res=1.391e-03 | BC=5.854e-05
    Adam   3000 | L=8.239e-04 | Res=6.245e-04 | BC=1.994e-05
    Adam   4000 | L=5.472e-04 | Res=4.119e-04 | BC=1.353e-05
    Adam   5000 | L=5.135e-04 | Res=3.874e-04 | BC=1.261e-05
    LBFGS   40 | L=1.564e-05 | Res=1.405e-05 | BC=1.595e-07
    LBFGS   80 | L=3.582e-06 | Res=2.937e-06 | BC=6.451e-08
    LBFGS   40 | L=1.564e-05 | Res=1.405e-05 | BC=1.595e-07
    LBFGS   80 | L=3.582e-06 | Res=2.937e-06 | BC=6.451e-08
    LBFGS  120 | L=2.808e-06 | Res=2.234e-06 | BC=5.734e-08
    LBFGS   80 | L=3.582e-06 | Res=2.937e-06 | BC=6.451e-08
    LBFGS  120 | L=2.808e-06 | Res=2.234e-06 | BC=5.734e-08
    LBFGS  120 | L=2.808e-06 | Res=2.234e-06 | BC=5.734e-08
    LBFGS  160 | L=2.808e-06 | Res=2.234e-06 | BC=5.734e-08
    LBFGS  200 | L=2.808e-06 | Res=2.234e-06 | BC=5.734e-08
    Done in 99.2s. Final loss: 2.808e-06
    Computing ||r||_{H⁻¹} (interior residual)...
    Computing ||∇w|| (BC lifting norm)...
    Computing ||r||_{L²}...
    Computing true errors...

    ──── RESULTS ────
    (1/α)||r||_{H⁻¹}   = 4.847130e-05
    ||∇w|| (BC lift)    = 1.560113e-03
    TOTAL estimate      = 1.608585e-03
    True H¹             = 1.699540e-03
    Effectivity η       = 0.9465
    Training loss       = 2.634476e-06
    BC error (L²∂Ω)     = 2.457203e-04

  === Poisson (smooth) | 5x128 ===
    Arch: [128, 128, 128, 128, 128], tanh, fourier=False, params=66561
    Adam   1000 | L=3.654e-02 | Res=2.462e-02 | BC=1.193e-03
    Adam   2000 | L=2.164e-02 | Res=2.069e-02 | BC=9.492e-05
    Adam   3000 | L=2.122e-03 | Res=1.736e-03 | BC=3.863e-05
    Adam   4000 | L=3.019e-04 | Res=2.517e-04 | BC=5.018e-06
    Adam   5000 | L=1.665e-04 | Res=1.490e-04 | BC=1.746e-06
    Adam   6000 | L=1.456e-04 | Res=1.327e-04 | BC=1.294e-06
    Adam   7000 | L=1.203e-04 | Res=1.093e-04 | BC=1.101e-06
    Adam   8000 | L=1.292e-04 | Res=1.181e-04 | BC=1.112e-06
    LBFGS   60 | L=4.152e-06 | Res=3.295e-06 | BC=8.579e-08
    LBFGS  120 | L=2.136e-06 | Res=1.720e-06 | BC=4.159e-08
    LBFGS  180 | L=2.136e-06 | Res=1.720e-06 | BC=4.159e-08
    LBFGS  240 | L=2.136e-06 | Res=1.720e-06 | BC=4.159e-08
    LBFGS  300 | L=2.136e-06 | Res=1.720e-06 | BC=4.159e-08
    Done in 559.5s. Final loss: 2.136e-06
    Computing ||r||_{H⁻¹} (interior residual)...
    Computing ||∇w|| (BC lifting norm)...
    Computing ||r||_{L²}...
    Computing true errors...

    ──── RESULTS ────
    (1/α)||r||_{H⁻¹}   = 5.078448e-05
    ||∇w|| (BC lift)    = 1.406955e-03
    TOTAL estimate      = 1.457740e-03
    True H¹             = 1.527760e-03
    Effectivity η       = 0.9542
    Training loss       = 2.169616e-06
    BC error (L²∂Ω)     = 2.096044e-04

██████████████████████████████████████████████████████████████████████
█  BENCHMARK 2: VARIABLE-COEFFICIENT DIFFUSION
██████████████████████████████████████████████████████████████████████

  === Variable coeff | 4x64 ===
    Arch: [64, 64, 64, 64], tanh, fourier=False, params=12737
    Adam   1000 | L=1.565e-01 | Res=9.932e-02 | BC=5.715e-03
    Adam   2000 | L=2.376e-02 | Res=1.311e-02 | BC=1.065e-03
    Adam   3000 | L=5.748e-03 | Res=3.324e-03 | BC=2.424e-04
    Adam   4000 | L=2.463e-03 | Res=1.719e-03 | BC=7.446e-05
    Adam   5000 | L=1.367e-03 | Res=9.007e-04 | BC=4.666e-05
    Adam   6000 | L=1.563e-03 | Res=1.283e-03 | BC=2.803e-05
    Adam   7000 | L=1.015e-03 | Res=6.588e-04 | BC=3.560e-05
    Adam   8000 | L=8.449e-04 | Res=5.460e-04 | BC=2.989e-05
    LBFGS   60 | L=1.540e-05 | Res=1.246e-05 | BC=2.942e-07
    LBFGS  120 | L=6.070e-06 | Res=4.902e-06 | BC=1.167e-07
    LBFGS  180 | L=6.070e-06 | Res=4.902e-06 | BC=1.167e-07
    LBFGS  240 | L=6.070e-06 | Res=4.902e-06 | BC=1.167e-07
    LBFGS  300 | L=6.070e-06 | Res=4.902e-06 | BC=1.167e-07
    Done in 327.4s. Final loss: 6.070e-06
    Computing ||r||_{H⁻¹} (interior residual)...
    Computing ||∇w|| (BC lifting norm)...
    Computing ||r||_{L²}...
    Computing true errors...

    ──── RESULTS ────
    (1/α)||r||_{H⁻¹}   = 1.263961e-04
    ||∇w|| (BC lift)    = 2.604432e-03
    TOTAL estimate      = 2.730828e-03
    True H¹             = 2.892525e-03
    Effectivity η       = 0.9441
    Training loss       = 5.593013e-06
    BC error (L²∂Ω)     = 3.717316e-04

  === Variable coeff | 4x128 ===
    Arch: [128, 128, 128, 128], tanh, fourier=False, params=50049
    Adam   1000 | L=6.708e-02 | Res=1.673e-02 | BC=5.036e-03
    Adam   2000 | L=9.049e-03 | Res=4.115e-03 | BC=4.934e-04
    Adam   3000 | L=6.796e-03 | Res=5.410e-03 | BC=1.386e-04
    Adam   4000 | L=9.483e-04 | Res=6.272e-04 | BC=3.211e-05
    Adam   5000 | L=5.185e-04 | Res=3.145e-04 | BC=2.040e-05
    Adam   6000 | L=2.015e-03 | Res=1.897e-03 | BC=1.181e-05
    Adam   7000 | L=5.502e-04 | Res=4.034e-04 | BC=1.468e-05
    Adam   8000 | L=3.330e-04 | Res=2.274e-04 | BC=1.055e-05
    LBFGS   60 | L=1.004e-05 | Res=6.534e-06 | BC=3.505e-07
    LBFGS  120 | L=5.323e-06 | Res=3.398e-06 | BC=1.926e-07
    LBFGS  180 | L=5.323e-06 | Res=3.398e-06 | BC=1.926e-07
    LBFGS  240 | L=5.323e-06 | Res=3.398e-06 | BC=1.926e-07
    LBFGS  300 | L=5.323e-06 | Res=3.398e-06 | BC=1.926e-07
    Done in 604.9s. Final loss: 5.323e-06
    Computing ||r||_{H⁻¹} (interior residual)...
    Computing ||∇w|| (BC lifting norm)...
    Computing ||r||_{L²}...
    Computing true errors...

    ──── RESULTS ────
    (1/α)||r||_{H⁻¹}   = 1.178357e-04
    ||∇w|| (BC lift)    = 3.025514e-03
    TOTAL estimate      = 3.143350e-03
    True H¹             = 3.263227e-03
    Effectivity η       = 0.9633
    Training loss       = 5.912790e-06
    BC error (L²∂Ω)     = 4.433967e-04

██████████████████████████████████████████████████████████████████████
█  BENCHMARK 3: L-SHAPED DOMAIN (CORNER SINGULARITY)
██████████████████████████████████████████████████████████████████████

  === L-shaped (singularity) | 4x64 ===
    Arch: [64, 64, 64, 64], tanh, fourier=False, params=12737
    Adam   1000 | L=2.975e-02 | Res=6.434e-03 | BC=2.331e-03
    Adam   2000 | L=9.396e-03 | Res=1.849e-03 | BC=7.547e-04
    Adam   3000 | L=7.128e-03 | Res=2.769e-03 | BC=4.359e-04
    Adam   4000 | L=2.771e-03 | Res=8.624e-04 | BC=1.909e-04
    Adam   5000 | L=3.643e-03 | Res=7.813e-04 | BC=2.862e-04
    Adam   6000 | L=3.134e-03 | Res=6.536e-04 | BC=2.481e-04
    Adam   7000 | L=2.494e-03 | Res=6.755e-04 | BC=1.818e-04
    Adam   8000 | L=3.758e-03 | Res=4.912e-04 | BC=3.267e-04
    LBFGS   60 | L=4.416e-04 | Res=1.516e-04 | BC=2.900e-05
    LBFGS  120 | L=2.177e-04 | Res=6.977e-05 | BC=1.480e-05
    LBFGS  180 | L=1.397e-04 | Res=4.514e-05 | BC=9.460e-06
    LBFGS  240 | L=8.198e-05 | Res=3.728e-05 | BC=4.470e-06
    LBFGS  300 | L=4.933e-05 | Res=2.850e-05 | BC=2.084e-06
    Done in 542.0s. Final loss: 4.933e-05
    Computing ||r||_{H⁻¹} (interior residual)...
    Computing ||∇w|| (BC lifting norm)...
    Computing ||r||_{L²}...
    Computing true errors...

    ──── RESULTS ────
    (1/α)||r||_{H⁻¹}   = 4.417395e-02
    ||∇w|| (BC lift)    = 1.365013e-03
    TOTAL estimate      = 4.553896e-02
    True H¹             = 2.929133e-02
    Effectivity η       = 1.5547
    Training loss       = 1.293660e+00
    BC error (L²∂Ω)     = 1.114769e-03

  === L-shaped (singularity) | 4x128 ===
    Arch: [128, 128, 128, 128], tanh, fourier=False, params=50049
    Adam   1000 | L=1.552e-02 | Res=3.143e-03 | BC=1.238e-03
    Adam   2000 | L=7.188e-03 | Res=1.810e-03 | BC=5.377e-04
    Adam   3000 | L=2.980e-03 | Res=1.269e-03 | BC=1.711e-04
    Adam   4000 | L=1.844e-03 | Res=5.413e-04 | BC=1.303e-04
    Adam   5000 | L=2.589e-03 | Res=4.307e-04 | BC=2.158e-04
    Adam   6000 | L=2.782e-03 | Res=4.346e-04 | BC=2.347e-04
    Adam   7000 | L=1.607e-03 | Res=4.710e-04 | BC=1.136e-04
    Adam   8000 | L=1.043e-03 | Res=4.158e-04 | BC=6.270e-05
    Adam   9000 | L=1.365e-03 | Res=3.194e-04 | BC=1.045e-04
    Adam  10000 | L=1.812e-03 | Res=2.362e-04 | BC=1.576e-04
    LBFGS  100 | L=4.197e-04 | Res=7.498e-05 | BC=3.447e-05
    LBFGS  200 | L=2.496e-04 | Res=4.974e-05 | BC=1.998e-05
    LBFGS  300 | L=1.460e-04 | Res=3.783e-05 | BC=1.082e-05
    LBFGS  400 | L=1.226e-04 | Res=3.146e-05 | BC=9.112e-06
    LBFGS  500 | L=1.226e-04 | Res=3.146e-05 | BC=9.112e-06
    Done in 1776.0s. Final loss: 1.226e-04
    Computing ||r||_{H⁻¹} (interior residual)...
    Computing ||∇w|| (BC lifting norm)...
    Computing ||r||_{L²}...
    Computing true errors...

    ──── RESULTS ────
    (1/α)||r||_{H⁻¹}   = 1.526932e-03
    ||∇w|| (BC lift)    = 2.273181e-03
    TOTAL estimate      = 3.800113e-03
    True H¹             = 5.032689e-02
    Effectivity η       = 0.0755
    Training loss       = 3.037703e-03
    BC error (L²∂Ω)     = 2.559347e-03

██████████████████████████████████████████████████████████████████████
█  BENCHMARK 4: CONVECTION-DOMINATED
██████████████████████████████████████████████████████████████████████

  === Convection (ε=0.1) | 4x64_eps01 ===
    Arch: [64, 64, 64, 64], tanh, fourier=False, params=12737
    Adam   1000 | L=4.827e-03 | Res=4.383e-03 | BC=2.219e-05
    Adam   2000 | L=2.749e-03 | Res=2.251e-03 | BC=2.489e-05
    Adam   3000 | L=1.762e-03 | Res=1.620e-03 | BC=7.116e-06
    Adam   4000 | L=1.313e-03 | Res=1.182e-03 | BC=6.532e-06
    Adam   5000 | L=1.285e-03 | Res=1.155e-03 | BC=6.528e-06
    LBFGS   40 | L=2.584e-05 | Res=1.639e-05 | BC=4.722e-07
    LBFGS   80 | L=1.750e-05 | Res=1.229e-05 | BC=2.604e-07
    LBFGS  120 | L=1.750e-05 | Res=1.229e-05 | BC=2.604e-07
    LBFGS  160 | L=1.750e-05 | Res=1.229e-05 | BC=2.604e-07
    LBFGS  200 | L=1.750e-05 | Res=1.229e-05 | BC=2.604e-07
    Done in 171.1s. Final loss: 1.750e-05
    Computing ||r||_{H⁻¹} (interior residual)...
    Computing ||∇w|| (BC lifting norm)...
    Computing ||r||_{L²}...
    Computing true errors...

    ──── RESULTS ────
    (1/α)||r||_{H⁻¹}   = 1.584516e-03
    ||∇w|| (BC lift)    = 4.033669e-03
    TOTAL estimate      = 5.618185e-03
    True H¹             = 6.242647e-01
    Effectivity η       = 0.0090
    Training loss       = 1.311858e-05
    BC error (L²∂Ω)     = 5.416329e-04

  === Convection (ε=0.01) | 4x64_eps001 ===
    Arch: [64, 64, 64, 64], tanh, fourier=False, params=12737
    Adam   1000 | L=3.770e-02 | Res=3.298e-02 | BC=9.441e-05
    Adam   2000 | L=2.676e-02 | Res=2.578e-02 | BC=1.947e-05
    Adam   3000 | L=1.728e-02 | Res=1.704e-02 | BC=4.831e-06
    Adam   4000 | L=1.402e-02 | Res=1.368e-02 | BC=6.788e-06
    Adam   5000 | L=1.221e-02 | Res=1.168e-02 | BC=1.061e-05
    Adam   6000 | L=1.057e-02 | Res=1.013e-02 | BC=8.910e-06
    Adam   7000 | L=9.622e-03 | Res=8.981e-03 | BC=1.282e-05
    Adam   8000 | L=9.571e-03 | Res=8.956e-03 | BC=1.231e-05
    LBFGS   60 | L=4.156e-05 | Res=3.297e-05 | BC=1.718e-07
    LBFGS  120 | L=1.749e-05 | Res=1.333e-05 | BC=8.331e-08
    LBFGS  180 | L=1.749e-05 | Res=1.333e-05 | BC=8.331e-08
    LBFGS  240 | L=1.749e-05 | Res=1.333e-05 | BC=8.331e-08
    LBFGS  300 | L=1.749e-05 | Res=1.333e-05 | BC=8.331e-08
    Done in 271.1s. Final loss: 1.749e-05
    Computing ||r||_{H⁻¹} (interior residual)...
    Computing ||∇w|| (BC lifting norm)...
    Computing ||r||_{L²}...
    Computing true errors...

    ──── RESULTS ────
    (1/α)||r||_{H⁻¹}   = 1.601888e-02
    ||∇w|| (BC lift)    = 2.752823e-03
    TOTAL estimate      = 1.877171e-02
    True H¹             = 4.266342e+00
    Effectivity η       = 0.0044
    Training loss       = 1.599404e-05
    BC error (L²∂Ω)     = 3.127225e-04

  === Convection (ε=0.01) | 4x128_eps001 ===
    Arch: [128, 128, 128, 128], tanh, fourier=False, params=50049
    Adam   1000 | L=4.310e-02 | Res=3.245e-02 | BC=2.131e-04
    Adam   2000 | L=3.342e-02 | Res=3.066e-02 | BC=5.538e-05
    Adam   3000 | L=2.353e-02 | Res=2.249e-02 | BC=2.074e-05
    Adam   4000 | L=1.613e-02 | Res=1.560e-02 | BC=1.049e-05
    Adam   5000 | L=1.299e-02 | Res=1.252e-02 | BC=9.520e-06
    Adam   6000 | L=1.103e-02 | Res=1.032e-02 | BC=1.425e-05
    Adam   7000 | L=9.265e-03 | Res=8.529e-03 | BC=1.471e-05
    Adam   8000 | L=7.546e-03 | Res=7.002e-03 | BC=1.088e-05
    Adam   9000 | L=7.576e-03 | Res=6.841e-03 | BC=1.468e-05
    Adam  10000 | L=7.388e-03 | Res=6.708e-03 | BC=1.359e-05
    LBFGS  100 | L=1.309e-05 | Res=1.196e-05 | BC=2.268e-08
    LBFGS  200 | L=1.027e-05 | Res=9.412e-06 | BC=1.720e-08
    LBFGS  300 | L=1.027e-05 | Res=9.412e-06 | BC=1.720e-08
    LBFGS  400 | L=1.027e-05 | Res=9.412e-06 | BC=1.720e-08
    LBFGS  500 | L=1.027e-05 | Res=9.412e-06 | BC=1.720e-08
    Done in 745.4s. Final loss: 1.027e-05
    Computing ||r||_{H⁻¹} (interior residual)...
    Computing ||∇w|| (BC lifting norm)...
    Computing ||r||_{L²}...
    Computing true errors...

    ──── RESULTS ────
    (1/α)||r||_{H⁻¹}   = 1.290291e-02
    ||∇w|| (BC lift)    = 1.158528e-03
    TOTAL estimate      = 1.406144e-02
    True H¹             = 4.232252e+00
    Effectivity η       = 0.0033
    Training loss       = 9.712426e-06
    BC error (L²∂Ω)     = 1.461720e-04

======================================================================
 GENERATING FIGURES
======================================================================
  Saved: D:\Work\PHD-2026\Project\project-2\pinn-error-bounds\figures\effectivity_v2.png
  Saved: D:\Work\PHD-2026\Project\project-2\pinn-error-bounds\figures\decomposition_v2.png
  Saved: D:\Work\PHD-2026\Project\project-2\pinn-error-bounds\figures\loss_vs_error_v2.png
  Saved: D:\Work\PHD-2026\Project\project-2\pinn-error-bounds\figures\solution_Poisson_4x64.png
  Saved: D:\Work\PHD-2026\Project\project-2\pinn-error-bounds\figures\solution_VarCoeff_4x64.png
  Saved: D:\Work\PHD-2026\Project\project-2\pinn-error-bounds\figures\solution_LShaped_4x64.png
  Saved: D:\Work\PHD-2026\Project\project-2\pinn-error-bounds\figures\solution_Conv_eps01.png


========================================================================================================================
 COMPLETE RESULTS TABLE (CORRECTED: Interior + BC Lifting)
========================================================================================================================
Experiment                  α   (1/α)||r||       ||∇w||    Total Est      True H¹        η         Loss
------------------------------------------------------------------------------------------------------------------------
Poisson_4x64            1.000   4.8471e-05   1.5601e-03   1.6086e-03   1.6995e-03   0.9465   2.6345e-06
Poisson_5x128           1.000   5.0784e-05   1.4070e-03   1.4577e-03   1.5278e-03   0.9542   2.1696e-06
VarCoeff_4x64           0.500   1.2640e-04   2.6044e-03   2.7308e-03   2.8925e-03   0.9441   5.5930e-06
VarCoeff_4x128          0.500   1.1784e-04   3.0255e-03   3.1433e-03   3.2632e-03   0.9633   5.9128e-06
LShaped_4x64            1.000   4.4174e-02   1.3650e-03   4.5539e-02   2.9291e-02   1.5547   1.2937e+00
LShaped_4x128           1.000   1.5269e-03   2.2732e-03   3.8001e-03   5.0327e-02   0.0755   3.0377e-03
Conv_eps01              0.100   1.5845e-03   4.0337e-03   5.6182e-03   6.2426e-01   0.0090   1.3119e-05
Conv_eps001             0.010   1.6019e-02   2.7528e-03   1.8772e-02   4.2663e+00   0.0044   1.5994e-05
Conv_eps001_deep        0.010   1.2903e-02   1.1585e-03   1.4061e-02   4.2323e+00   0.0033   9.7124e-06

  Results: D:\Work\PHD-2026\Project\project-2\pinn-error-bounds\results/results_v2.json
  Figures: D:\Work\PHD-2026\Project\project-2\pinn-error-bounds\figures/