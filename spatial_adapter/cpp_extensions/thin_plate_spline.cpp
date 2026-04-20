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
    const arma::mat& Phi)
{
    if (original_location.n_rows != Phi.n_rows) {
        throw std::runtime_error("Mismatch: Phi.n_rows = " + std::to_string(Phi.n_rows) +
                         ", expected " + std::to_string(original_location.n_rows));
    }
    if (new_location.n_cols != original_location.n_cols) {
        throw std::runtime_error("Mismatch: new_location.n_cols = " + std::to_string(new_location.n_cols) +
                         ", expected " + std::to_string(original_location.n_cols));
    }
    int p = original_location.n_rows, d = original_location.n_cols, K = Phi.n_cols;
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
        for (int k = 0; k < d; ++k)
            L(i, p + k + 1) = original_location(i, k);
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
            for (int k = 0; k < d; ++k)
                poly += para(p + k + 1, i) * new_location(new_i, k);

            eigen_fn(new_i, i) = psum + poly;
        }
    }

    return eigen_fn;
}
// -----------------------------------------------------------------------------
// Helper: Compute the estimated covariance (rank-K) in the phi-basis
// -----------------------------------------------------------------------------
arma::mat computeEstimatedCovarianceMulti(
    const arma::mat& phi,               // p x K basis
    const arma::mat& V,                 // K x K subspace eigenvectors
    const arma::vec& lambda_trunc,      // K top eigenvalues
    const arma::mat& predicted_phi      // p' x K predicted basis
) {
    return phi * V * arma::diagmat(lambda_trunc) * (predicted_phi * V).t();
}

// -----------------------------------------------------------------------------
// Helper: Compute spatial predictions (fixed-rank kriging)
// -----------------------------------------------------------------------------
arma::mat computeSpatialPredictionsMulti(
    const arma::mat& phi,               // p x K basis
    const arma::mat& V,                 // K x K subspace eigenvectors
    const arma::vec& lambda_trunc,      // K top eigenvalues
    double noise_var,                   // noise variance
    const arma::mat& Y                  // n x p centered residuals
) {
    arma::vec weights = lambda_trunc / (lambda_trunc + noise_var);
    return Y * phi * V * arma::diagmat(weights) * V.t() * phi.t();
}

// -----------------------------------------------------------------------------
// Function: estimateCovariance
//   Estimates top-K eigenvalues and noise variance from training residuals
// -----------------------------------------------------------------------------
py::dict estimateCovariance(
    const arma::mat& phi,    // p × K training basis
    const arma::mat& Y       // n × p centered residuals
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

    const int n = Y.n_rows;     // number of samples
    const int p = phi.n_rows;   // original dimension
    const int K = phi.n_cols;   // basis dimension

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

    // 3) Determine number of components to keep
    int KK = std::min(K, static_cast<int>(sorted_vals.n_elem));
    if (KK < 1) {
        std::cerr << "estimateCovariance: warning – no eigenvalues found; "
                     "forcing KK = 1\n";
        KK = 1;
    }
    arma::vec top_eigs = sorted_vals.head(KK);

    // 4) Compute isotropic noise variance
    double noise_var = 0.0;
    if (KK < p) {
        double sum_top = arma::sum(top_eigs);
        noise_var = std::max(0.0, (total_var - sum_top) / (p - KK));
    }

    // 5) Reconstruct low-rank covariance estimate
    arma::mat V       = eigvec.cols(idx.head(KK)); // K × KK
    arma::mat proj    = phi * V;                   // p × KK
    arma::mat est_cov = proj * arma::diagmat(top_eigs) * proj.t(); // p × p

    // Return results to Python
    return py::dict(
        "eigenvalues"_a          = py::array_t<double>(top_eigs.n_elem, top_eigs.memptr()),
        "V"_a                    = py::array_t<double>({V.n_rows, V.n_cols}, V.memptr()),
        "noise_var"_a            = noise_var,
        "estimated_covariance"_a = py::array_t<double>(
                                      {est_cov.n_rows, est_cov.n_cols},
                                      est_cov.memptr())
    );
}

// -----------------------------------------------------------------------------
// Fixed-rank kriging: uses learned basis parameters and training residuals to predict at new locations
// -----------------------------------------------------------------------------
 py::dict fixedRankKriging(
    const arma::mat& phi_train,   // p × K training basis
    const arma::mat& V,           // K × KK eigenvector matrix from estimateCovariance
    const arma::vec& lambda,      // KK top eigenvalues
    double noise_var,             // noise variance from estimateCovariance
    const arma::mat& R_train,     // n × p centered residuals at training sites
    const arma::mat& phi_pred     // p* × K basis at prediction sites
) {
    // 1) Validate inputs
    if (phi_train.empty() || V.empty() || R_train.empty() || phi_pred.empty()) {
        throw std::invalid_argument{
            "fixedRankKriging: all input matrices must be non-empty"
        };
    }
    int p   = phi_train.n_rows;      // original dimension
    int K   = phi_train.n_cols;      // basis dimension
    int KK  = V.n_cols;              // reduced-rank dimension
    int n   = R_train.n_rows;        // number of training samples
    int p_s = phi_pred.n_rows;       // number of prediction sites

    // Check dimensions
    if ((int)V.n_rows      != K ||
        (int)lambda.n_elem != KK ||
        R_train.n_cols     != p ||
        phi_pred.n_cols    != K)
    {
        throw std::invalid_argument{
            "fixedRankKriging: dimension mismatch among phi_train, V, lambda, R_train, or phi_pred"
        };
    }

    // 2) Compute kriging weights: w_i = λ_i / (λ_i + noise_var)
    arma::vec w = lambda / (lambda + noise_var);  // length KK

    // 3) Project residuals into reduced subspace:   U = R_train * phi_train * V   (n × KK)
    arma::mat U = R_train * phi_train * V;

    // 4) Weight each component:   W = U * diag(w)   (n × KK)
    arma::mat W = U * arma::diagmat(w);

    // 5) Prepare prediction basis in reduced subspace:
    //    B = Vᵀ * phi_predᵀ    (KK × p*)
    arma::mat B = V.t() * phi_pred.t();

    // 6) Compute spatial predictions:   ΔY* = W * B   (n × p*)
    arma::mat spatial_pred = W * B;

    // 7) Return results to Python
    return py::dict(
        "spatial_predictions"_a = py::array_t<double>(
            { spatial_pred.n_rows, spatial_pred.n_cols },
            spatial_pred.memptr()
        )
    );
}


static arma::mat np_to_arma_mat(py::array_t<double>& array) {
    auto buf = array.request();
    if (buf.ndim < 1 || buf.ndim > 2)
        throw std::runtime_error("NumPy array must be 1D or 2D");
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


PYBIND11_MODULE(thin_plate_spline, m) {
    m.doc() = "Thin Plate Spline Module";

    m.def(
        "smoothing_penalty_matrix",
        [](py::array_t<double> location) -> py::array_t<double> {
            py::buffer_info buf = location.request();
            if (buf.ndim < 1 || buf.ndim > 2) {
                throw std::runtime_error("Input must be a 1D or 2D NumPy array.");
            }

            // Create Armadillo matrix with proper dimensions
            arma::mat loc(buf.shape[0],
                         (buf.ndim == 1) ? 1 : buf.shape[1]);

            // Copy data with proper layout
            double* ptr = static_cast<double*>(buf.ptr);
            for (size_t i = 0; i < loc.n_rows; ++i) {
                for (size_t j = 0; j < loc.n_cols; ++j) {
                    loc(i, j) = ptr[i * loc.n_cols + j];
                }
            }

            // Compute the smoothing penalty matrix
            arma::mat result = computeSmoothingPenaltyMatrix(loc);

            // Convert back to NumPy array
            return py::array_t<double>(
                {result.n_rows, result.n_cols},  // Shape of the matrix
                result.memptr()                  // Pointer to the data
            );
        },
        "Generate a smoothing penalty matrix for 1D, 2D, or 3D data"
    );

    m.def(
        "interpolate_eigenfunction",
        [](py::array_t<double> new_loc, py::array_t<double> orig_loc, py::array_t<double> phi) -> py::array_t<double> {
            // Helper function to safely convert NumPy array to Armadillo matrix
            auto convert_to_arma = [](py::array_t<double>& array) {
                arma::mat result(array.request().shape[0],
                               array.request().ndim == 1 ? 1 : array.request().shape[1]);
                double* ptr = static_cast<double*>(array.request().ptr);
                for (size_t i = 0; i < result.n_rows; ++i) {
                    for (size_t j = 0; j < result.n_cols; ++j) {
                        result(i, j) = ptr[i * result.n_cols + j];
                    }
                }
                return result;
            };

            // Convert all arrays using the helper function
            arma::mat new_location = convert_to_arma(new_loc);
            arma::mat original_location = convert_to_arma(orig_loc);
            arma::mat Phi = convert_to_arma(phi);

            arma::mat result = interpolateEigenFunction(new_location, original_location, Phi);

            // Convert back to NumPy array
            return py::array_t<double>(
                {result.n_rows, result.n_cols},  // Shape of the matrix
                result.memptr()                  // Pointer to the data
            );
        },
        "Interpolate thin-plate spline basis at new locations"
    );

    m.def(
        "estimate_covariance",
        [](py::array_t<double> phi_arr, py::array_t<double> Y_arr) {
            arma::mat phi = np_to_arma_mat(phi_arr);
            arma::mat Y   = np_to_arma_mat(Y_arr);
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
            arma::mat V         = np_to_arma_mat(V_arr);
            arma::vec lambda    = np_to_arma_mat(lambda_arr);
            arma::mat Y_new     = np_to_arma_mat(Y_new_arr);
            arma::mat phi_pred  = np_to_arma_mat(phi_pred_arr);

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
