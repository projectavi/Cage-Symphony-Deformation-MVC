//
//  StructuralAffinityController.hpp
//

#ifndef StructuralAffinityController_hpp
#define StructuralAffinityController_hpp

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>

using namespace Eigen;
using namespace std;

class StructuralAffinityController {
private:
    SparseMatrix<double, RowMajor> affinity_;
    int num_vertices_;
    double epsilon_;
    bool ready_;

public:
    StructuralAffinityController();
    void BuildFromWeights(const MatrixXd& lower_to_upper_weights, double epsilon);
    void ApplyDeltasWithPinnedSources(const vector<int>& source_ids,
                                      const vector<Vector3d>& source_deltas,
                                      const MatrixXd& base_cage,
                                      MatrixXd& out_cage,
                                      double neighbor_alpha = 1.0) const;
    bool IsReady() const;
    int GetNumVertices() const;
    int GetNumNonZeros() const;
    double GetDensity() const;
    const SparseMatrix<double, RowMajor>& GetAffinity() const;
};

#endif /* StructuralAffinityController_hpp */
