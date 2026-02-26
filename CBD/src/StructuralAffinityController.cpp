//
//  StructuralAffinityController.cpp
//

#include "StructuralAffinityController.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

static const double kAffinityNumericalEpsilon = 1e-12;

StructuralAffinityController::StructuralAffinityController() {
    num_vertices_ = 0;
    epsilon_ = 0.;
    ready_ = false;
}

void StructuralAffinityController::BuildFromWeights(const MatrixXd& lower_to_upper_weights,
                                                    double epsilon,
                                                    AffinityMode mode) {
    // Precompute sparse affinity C from lower->upper weights.
    // Min-weight mode:
    //   C(i,j) = sum_k min(w_i_k, w_j_k) / sum_k w_i_k
    // Analytical mode:
    //   C(i,j) = sum_k (w_i_k * w_j_k) / sum_k (w_i_k^2)
    num_vertices_ = lower_to_upper_weights.rows();
    epsilon_ = max(0., epsilon);
    ready_ = false;

    affinity_.resize(num_vertices_, num_vertices_);
    affinity_.setZero();

    if (num_vertices_ == 0) {
        ready_ = true;
        return;
    }

    vector<VectorXd> source_weights_rows(num_vertices_);
    VectorXd denominators = VectorXd::Zero(num_vertices_);

    // Min-weight mode uses non-negative weights to preserve conservative overlap semantics.
    // Analytical mode uses raw weights directly in dot products.
    for (int i = 0; i < num_vertices_; ++i) {
        if (mode == AffinityMode::MinWeightIntersection) {
            source_weights_rows[i] = lower_to_upper_weights.row(i).cwiseMax(0.0);
            denominators(i) = source_weights_rows[i].sum();
        } else {
            source_weights_rows[i] = lower_to_upper_weights.row(i);
            denominators(i) = source_weights_rows[i].squaredNorm();
        }
    }

    vector<Triplet<double>> triplets;
    triplets.reserve(num_vertices_ * 8);

    for (int i = 0; i < num_vertices_; ++i) {
        // The source vertex always follows 100% of user drag.
        triplets.push_back(Triplet<double>(i, i, 1.0));

        if (denominators(i) <= kAffinityNumericalEpsilon) {
            continue;
        }

        const VectorXd& wi = source_weights_rows[i];

        for (int j = 0; j < num_vertices_; ++j) {
            if (j == i) {
                continue;
            }

            const VectorXd& wj = source_weights_rows[j];
            double numerator = 0.0;
            if (mode == AffinityMode::MinWeightIntersection) {
                numerator = wi.cwiseMin(wj).sum();
            } else {
                numerator = wi.dot(wj);
            }

            double cij = numerator / denominators(i);
            if (mode == AffinityMode::MinWeightIntersection) {
                cij = min(1.0, max(0.0, cij));
            }

            // Keep entries above threshold; drop tiny numerical noise.
            if (cij >= epsilon_ && cij > kAffinityNumericalEpsilon) {
                triplets.push_back(Triplet<double>(i, j, cij));
            }
        }
    }

    affinity_.setFromTriplets(
        triplets.begin(),
        triplets.end(),
        [](const double& a, const double& b) { return max(a, b); });
    affinity_.makeCompressed();
    ready_ = true;
}

void StructuralAffinityController::ApplyDeltasWithPinnedSources(
    const vector<int>& source_ids,
    const vector<Vector3d>& source_deltas,
    const MatrixXd& base_cage,
    MatrixXd& out_cage,
    double neighbor_alpha) const {
    out_cage = base_cage;
    if (!ready_ || base_cage.rows() != num_vertices_ || base_cage.cols() != 3) {
        return;
    }

    // 1) Additive propagation from all selected sources.
    MatrixXd accumulated_delta = MatrixXd::Zero(num_vertices_, 3);
    const int num_sources = min((int)source_ids.size(), (int)source_deltas.size());

    for (int source_i = 0; source_i < num_sources; ++source_i) {
        const int source_id = source_ids[source_i];
        if (source_id < 0 || source_id >= num_vertices_) {
            continue;
        }

        const Vector3d& source_delta = source_deltas[source_i];
        for (SparseMatrix<double, RowMajor>::InnerIterator it(affinity_, source_id); it; ++it) {
            const int target_id = it.col();
            accumulated_delta.row(target_id) += source_delta.transpose() * it.value();
        }
    }

    const double clamped_alpha = min(1.0, max(0.0, neighbor_alpha));
    accumulated_delta *= clamped_alpha;

    // 2) Hard-pin source handles to exact user drag after accumulation.
    for (int source_i = 0; source_i < num_sources; ++source_i) {
        const int source_id = source_ids[source_i];
        if (source_id < 0 || source_id >= num_vertices_) {
            continue;
        }
        accumulated_delta.row(source_id) = source_deltas[source_i];
    }

    // 3) Apply the final delta field to the base cage.
    out_cage += accumulated_delta;
}

bool StructuralAffinityController::IsReady() const {
    return ready_;
}

int StructuralAffinityController::GetNumVertices() const {
    return num_vertices_;
}

int StructuralAffinityController::GetNumNonZeros() const {
    return affinity_.nonZeros();
}

double StructuralAffinityController::GetDensity() const {
    if (num_vertices_ == 0) {
        return 0.;
    }
    return (double) affinity_.nonZeros() / (double) (num_vertices_ * num_vertices_);
}

const SparseMatrix<double, RowMajor>& StructuralAffinityController::GetAffinity() const {
    return affinity_;
}
