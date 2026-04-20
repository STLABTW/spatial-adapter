#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include <cmath>
#include <iostream>

namespace py = pybind11;
using namespace py::literals;
using namespace arma;

constexpr double EPSILON = 1e-8;

inline double computeDistance(const arma::rowvec& p1, const arma::rowvec& p2, int d) {
    if (d == 1) {
        return std::abs(p1(0) - p2(0));
    } else if (d == 2) {
        return std::hypot(p1(0) - p2(0), p1(1) - p2(1));
    } else if (d == 3) {
        return std::sqrt(std::pow(p1(0) - p2(0), 2) +
                         std::pow(p1(1) - p2(1), 2) +
                         std::pow(p1(2) - p2(2), 2));
    }
    throw std::invalid_argument("Unsupported dimension (only 1D, 2D, or 3D supported).");
}

inline double thinPlateSplineKernel(double r, int d) {
    if (r < EPSILON) return 0.0;

    if (d == 1)
        return std::pow(r, 3) / 12.0;
    else if (d == 2)
        return r * r * std::log(r) / (8.0 * datum::pi);
    else if (d == 3)
        return -r / (8.0 * datum::pi);
    else
        throw std::invalid_argument("Unsupported dimension (only 1D, 2D, or 3D supported).");
}

static arma::mat np_to_arma_mat(py::array_t<double>& array) {
    auto buf = array.request();
    if (buf.ndim < 1 || buf.ndim > 2) {
        throw std::runtime_error("NumPy array must be 1D or 2D");
    }

    size_t rows = buf.shape[0];
    size_t cols = (buf.ndim == 2 ? buf.shape[1] : 1);

    arma::mat M(rows, cols);
    double* ptr = static_cast<double*>(buf.ptr);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            M(i, j) = ptr[i * cols + j];
        }
    }
    return M;
}

static py::array_t<double> arma_mat_to_numpy(const arma::mat& M) {
    py::array_t<double> out(
        {static_cast<py::ssize_t>(M.n_rows), static_cast<py::ssize_t>(M.n_cols)}
    );

    auto buf = out.request();
    double* ptr = static_cast<double*>(buf.ptr);

    for (size_t i = 0; i < M.n_rows; ++i) {
        for (size_t j = 0; j < M.n_cols; ++j) {
            ptr[i * M.n_cols + j] = M(i, j);
        }
    }

    return out;
}

static py::array_t<double> arma_vec_to_numpy(const arma::vec& v) {
    py::array_t<double> out(static_cast<py::ssize_t>(v.n_elem));

    auto buf = out.request();
    double* ptr = static_cast<double*>(buf.ptr);

    for (size_t i = 0; i < v.n_elem; ++i) {
        ptr[i] = v(i);
    }

    return out;
}

arma::mat computeSmoothingPenaltyMatrix(const arma::mat& location) {
    int p = location.n_rows, d = location.n_cols;
    if (d < 1 || d > 3) {
        throw std::invalid_argument("Unsupported dimension (only 1D, 2D, or 3D supported).");
    }
    int total_size = p + d;
    mat L, Lp, Ip;

    L.zeros(total_size + 1, total_size + 1);
    Ip.eye(total_size + 1, total_size + 1);

    for (int i = 0; i < p; ++i) {
        for (int j = i + 1; j < p; ++j) { // Upper triangle
            double r = computeDistance(location.row(i), location.row(j), d);
            L(i, j) = thinPlateSplineKernel(r, d);
        }

        L(i, p) = 1.0;
        for (int k = 0; k < d; ++k) {
            L(i, p + k + 1) = location(i, k);
        }
    }

    L = symmatu(L);
    Lp = inv(L + EPSILON * Ip);
    Lp.shed_cols(p, total_size);
    Lp.shed_rows(p, total_size);
    L.shed_cols(p, total_size);
    L.shed_rows(p, total_size);

    return Lp.t() * (L * Lp);
}

arma::mat interpolateEigenFunction(
    const arma::mat& new_location,
    const arma::mat& original_location,
    const arma::mat& Phi
) {
    if (original_location.n_rows != Phi.n_rows) {
        throw std::runtime_error(
            "Mismatch: Phi.n_rows = " + std::to_string(Phi.n_rows) +
            ", expected " + std::to_string(original_location.n_rows)
        );
    }
    if (new_location.n_cols != original_location.n_cols) {
        throw std::runtime_error(
            "Mismatch: new_location.n_cols = " + std::to_string(new_location.n_cols) +
            ", expected " + std::to_string(original_location.n_cols)
        );
    }

    int p = original_location.n_rows;
    int d = original_location.n_cols;
    int K = Phi.n_cols;

    if (d < 1 || d > 3) {
        throw std::invalid_argument("Unsupported dimension (only 1D, 2D, or 3D supported).");
    }

    int total_size = p + d;

    // Step 1: Build L matrix
    arma::mat L(total_size + 1, total_size + 1, arma::fill::zeros);
    for (int i = 0; i < p; ++i) {
        for (int j = i + 1; j < p; ++j) {
            double r = computeDistance(original_location.row(i), original_location.row(j), d);
            L(i, j) = thinPlateSplineKernel(r, d);
        }

        L(i, p) = 1.0;
        for (int k = 0; k < d; ++k) {
            L(i, p + k + 1) = original_location(i, k);
        }
    }

    L = symmatu(L);

    // Step 2: Solve L * para = Phi_star
    arma::mat Phi_star(total_size + 1, K, arma::fill::zeros);
    Phi_star.rows(0, p - 1) = Phi;

    const arma::mat eye_L = arma::eye<arma::mat>(L.n_rows, L.n_cols);
    arma::mat para = arma::solve(L + EPSILON * eye_L, Phi_star);

    // Step 3: Compute interpolated values
    int pnew = new_location.n_rows;
    arma::mat eigen_fn(pnew, K, arma::fill::zeros);

    for (int new_i = 0; new_i < pnew; ++new_i) {
        for (int i = 0; i < K; ++i) {
            double psum = 0.0;
            for (int j = 0; j < p; ++j) {
                double r = computeDistance(new_location.row(new_i), original_location.row(j), d);
                if (r < EPSILON) continue;
                psum += para(j, i) * thinPlateSplineKernel(r, d);
            }

            double poly = para(p, i);  // Intercept
            for (int k = 0; k < d; ++k) {
                poly += para(p + k + 1, i) * new_location(new_i, k);
            }

            eigen_fn(new_i, i) = psum + poly;
        }
    }

    return eigen_fn;
}

// -----------------------------------------------------------------------------
// Function: estimateCovariance
//   Estimates top-K eigenvalues and noise variance from training residuals
// -----------------------------------------------------------------------------
py::dict estimateCovariance(
    const arma::mat& phi,
    const arma::mat& Y
) {
    // Validate inputs
    if (phi.n_rows == 0 || Y.n_rows == 0) {
        throw std::invalid_argument{
            "estimateCovariance: phi and Y must be non-empty"
        };
    }
    if (phi.n_rows != Y.n_cols) {
        throw std::invalid_argument{
            "estimateCovariance: phi.n_rows must equal Y.n_cols"
        };
    }

    const int n = Y.n_rows;
    const int p = phi.n_rows;
    const int K = phi.n_cols;

    // 1) Empirical covariance (p × p)
    arma::mat cov = (Y.t() * Y) / static_cast<double>(n);
    double total_var = arma::trace(cov);

    // 2) PCA in the φ-subspace (K × K)
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, phi.t() * cov * phi);

    // Sort eigenvalues in descending order
    arma::uvec idx = arma::sort_index(eigval, "descend");
    arma::vec sorted_vals = eigval(idx);

    // Upper bound on the number of retained components
    int KK = std::min(K, static_cast<int>(sorted_vals.n_elem));
    if (KK < 1) {
        std::cerr << "estimateCovariance: warning – no eigenvalues found; forcing KK = 1\n";
        KK = 1;
    }

    arma::vec top_eigs = sorted_vals.head(KK);

    // -------------------------------------------------------------------------
    // 3) Rank selection (Wang 2017 + empty-branch fix from Appendix B)
    //
    //    Iterate L from 1 to KK and compute, at rank L,
    //        sigma_L^2 = max(0, (tr(S) - sum_{k=1}^L d_k) / (p - L))
    //    the average residual variance if the signal subspace has rank L.
    //    Wang's rule keeps the largest L for which d_L > sigma_L^2; if no such
    //    L exists, we fall back to the empty branch:
    //        L_hat  = 0
    //        sigma^2 = tr(S) / p          (pure isotropic estimate)
    //        est_cov = sigma^2 * I        (no retained components)
    //    This matches eq:rank-fixed / eq:sigma-fixed in the paper's
    //    Appendix B and handles the weak-signal corner case where the
    //    original Wang rule leaves the defining set empty.  Smoothness
    //    penalty (tau) is applied upstream in the Python basis step and is
    //    not subtracted again here.
    // -------------------------------------------------------------------------
    int L_hat = 0;
    double sigma_hat = total_var / static_cast<double>(p);  // empty-branch default

    double cumsum = 0.0;
    for (int L = 1; L <= KK; ++L) {
        cumsum += top_eigs(L - 1);
        if (p - L <= 0) {
            // Degenerate full-rank case: cannot form sigma_L^2 at this L.
            break;
        }
        double sigma_L_squared = std::max(
            0.0,
            (total_var - cumsum) / static_cast<double>(p - L)
        );
        if (top_eigs(L - 1) > sigma_L_squared) {
            L_hat = L;
            sigma_hat = sigma_L_squared;
        }
    }

    // 4) Build shrunk eigenvalues: first L_hat entries noise-subtracted,
    //    remaining entries (if any) set to zero.  Keeping the output length
    //    equal to KK preserves backward compatibility with callers that
    //    expect `eigenvalues` to have K entries.
    arma::vec lambda_adj(KK, arma::fill::zeros);
    for (int k = 0; k < L_hat; ++k) {
        lambda_adj(k) = std::max(top_eigs(k) - sigma_hat, 0.0);
    }

    // 5) Reconstruct low-rank-plus-noise covariance estimate.
    //    matches eq:cov-est in the paper:
    //        Sigma_r = phi * Lambda * phi^T + sigma^2 * I
    arma::mat V = eigvec.cols(idx.head(KK));
    arma::mat proj = phi * V;
    arma::mat est_cov =
        proj * arma::diagmat(lambda_adj) * proj.t() +
        sigma_hat * arma::eye<arma::mat>(p, p);

    // Return results to Python.  `effective_rank` is a new field exposing
    // the rank selected by Wang's rule so downstream code can inspect it
    // without recomputing; existing fields keep their pre-fix shape.
    return py::dict(
        "eigenvalues"_a          = arma_vec_to_numpy(lambda_adj),
        "V"_a                    = arma_mat_to_numpy(V),
        "noise_var"_a            = sigma_hat,
        "estimated_covariance"_a = arma_mat_to_numpy(est_cov),
        "effective_rank"_a       = L_hat
    );
}

// -----------------------------------------------------------------------------
// Fixed-rank kriging: uses learned basis parameters and training residuals to predict at new locations
// -----------------------------------------------------------------------------
py::dict fixedRankKriging(
    const arma::mat& phi_train,
    const arma::mat& V,
    const arma::vec& lambda,
    double noise_var,
    const arma::mat& R_train,
    const arma::mat& phi_pred
) {
    // 1) Validate inputs
    if (phi_train.empty() || V.empty() || R_train.empty() || phi_pred.empty()) {
        throw std::invalid_argument{
            "fixedRankKriging: all input matrices must be non-empty"
        };
    }

    int p = phi_train.n_rows;
    int K = phi_train.n_cols;
    int KK = V.n_cols;
    int p_pred = phi_pred.n_rows;

    // Check dimensions
    if ((int)V.n_rows != K ||
        (int)lambda.n_elem != KK ||
        R_train.n_cols != p ||
        phi_pred.n_cols != K)
    {
        throw std::invalid_argument{
            "fixedRankKriging: dimension mismatch among phi_train, V, lambda, R_train, or phi_pred"
        };
    }

    arma::mat B_obs = phi_train * V;
    arma::mat B_pred = phi_pred * V;
    arma::mat Lambda = arma::diagmat(lambda);

    arma::mat Sigma_oo =
        B_obs * Lambda * B_obs.t() +
        noise_var * arma::eye<arma::mat>(p, p);

    arma::mat Sigma_op =
        B_obs * Lambda * B_pred.t();

    arma::mat Sigma_pp =
        B_pred * Lambda * B_pred.t() +
        noise_var * arma::eye<arma::mat>(p_pred, p_pred);

    arma::mat K_op = arma::solve(
        Sigma_oo,
        Sigma_op,
        arma::solve_opts::likely_sympd
    );

    arma::mat spatial_pred = R_train * K_op;

    arma::mat cond_cov = Sigma_pp - Sigma_op.t() * K_op;
    cond_cov = 0.5 * (cond_cov + cond_cov.t());

    arma::vec pred_var = arma::clamp(
        cond_cov.diag(),
        0.0,
        arma::datum::inf
    );

    // 7) Return results to Python
    return py::dict(
        "spatial_predictions"_a = arma_mat_to_numpy(spatial_pred),
        "predictive_variance"_a = arma_vec_to_numpy(pred_var)
    );
}

PYBIND11_MODULE(spatial_utils, m) {
    m.doc() = "Spatial Utilities Module: Eigenfunction covariance estimation, spatial prediction, and thin-plate spline penalty.";

    m.def(
        "smoothing_penalty_matrix",
        [](py::array_t<double> location) -> py::array_t<double> {
            arma::mat loc = np_to_arma_mat(location);
            arma::mat result = computeSmoothingPenaltyMatrix(loc);
            return arma_mat_to_numpy(result);
        },
        "Generate a smoothing penalty matrix for 1D, 2D, or 3D data"
    );

    m.def(
        "interpolate_eigenfunction",
        [](py::array_t<double> new_loc,
           py::array_t<double> orig_loc,
           py::array_t<double> phi) -> py::array_t<double> {
            arma::mat new_location = np_to_arma_mat(new_loc);
            arma::mat original_location = np_to_arma_mat(orig_loc);
            arma::mat Phi = np_to_arma_mat(phi);

            arma::mat result = interpolateEigenFunction(new_location, original_location, Phi);
            return arma_mat_to_numpy(result);
        },
        "Interpolate thin-plate spline basis at new locations"
    );

    m.def(
        "estimate_covariance",
        [](py::array_t<double> phi_arr, py::array_t<double> Y_arr) {
            arma::mat phi = np_to_arma_mat(phi_arr);
            arma::mat Y = np_to_arma_mat(Y_arr);
            return estimateCovariance(phi, Y);
        },
        "Compute top-K eigenvalues and noise variance from training residuals",
        py::arg("phi"), py::arg("Y")
    );

    m.def(
        "fixed_rank_kriging",
        [](py::array_t<double> phi_train_arr,
           py::array_t<double> V_arr,
           py::array_t<double> lambda_arr,
           double noise_var,
           py::array_t<double> Y_new_arr,
           py::array_t<double> phi_pred_arr) {

            arma::mat phi_train = np_to_arma_mat(phi_train_arr);
            arma::mat V = np_to_arma_mat(V_arr);
            arma::vec lambda = arma::vectorise(np_to_arma_mat(lambda_arr));
            arma::mat Y_new = np_to_arma_mat(Y_new_arr);
            arma::mat phi_pred = np_to_arma_mat(phi_pred_arr);

            return fixedRankKriging(
                phi_train,
                V,
                lambda,
                noise_var,
                Y_new,
                phi_pred
            );
        },
        "Apply learned basis + covariance to new centered residuals for spatial prediction",
        py::arg("phi_train"),
        py::arg("V"),
        py::arg("lambda"),
        py::arg("noise_var"),
        py::arg("Y_new"),
        py::arg("phi_pred")
    );
}
