//
//  DeformCageViewerPlugin.hpp
//  3DA_project_CageBasedDef_bin
//
//  Created by Benjamin Barral on 17/05/2019.
//

// Class for handling user interaction with the cage

#ifndef DeformCageViewerPlugin_hpp
#define DeformCageViewerPlugin_hpp

#include <stdio.h>

#include <algorithm>
#include <cfloat>
#include <vector>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/unproject_ray.h>

#include "MeanValueCoordController.hpp"
#include "StructuralAffinityController.hpp"

using namespace igl::opengl::glfw;
using namespace Eigen;
using namespace std;

static const double MAX_EDGE_SUM_DISTANCES_RATIO_ = 3.;

class DeformCageViewerPlugin : public ViewerPlugin {
private:
    MatrixXd V_cage_, V_cage_deformed_, C_points_cage_;
    MatrixXi F_cage_;
    int num_vertices_, num_faces_;
    int viewer_data_cage_id_, viewer_data_mesh_id_;
    int current_selected_point_id_, current_joint_selected_id_;
    bool is_selecting_point_, is_moving_point_, is_selecting_joint_point_, joint_point_selected_;
    double z_selected_point_;
    MeanValueCoordController mVCoord_controller_;
    StructuralAffinityController structural_affinity_controller_;
    double affinity_neighbor_alpha_;
    bool use_structural_affinity_mode_;
    Vector3d joint_point_;
    vector<vector<int>> neighbors_;
    Vector3d initial_mouse_world_pos_;
    vector<int> points_to_move_;
    VectorXi visited_points_;

    vector<int> active_source_ids_;
    VectorXi active_source_mask_;
    MatrixXd drag_base_cage_;
    vector<int> drag_source_ids_;

    void SyncViewerCage() {
        if (viewer == nullptr) {
            return;
        }
        viewer->data_list[viewer_data_cage_id_].set_mesh(V_cage_deformed_, F_cage_);
        viewer->data_list[viewer_data_cage_id_].set_points(V_cage_deformed_, C_points_cage_);
    }

    void ClearActiveSources() {
        active_source_ids_.clear();
        active_source_mask_ = VectorXi::Zero(num_vertices_);
    }

    bool IsSourceActive(const int& vertex_id) const {
        return vertex_id >= 0 && vertex_id < num_vertices_ &&
               active_source_mask_.size() == num_vertices_ && active_source_mask_(vertex_id) == 1;
    }

    void AddActiveSource(const int& vertex_id) {
        if (vertex_id < 0 || vertex_id >= num_vertices_ || IsSourceActive(vertex_id)) {
            return;
        }
        active_source_mask_(vertex_id) = 1;
        active_source_ids_.push_back(vertex_id);
    }

    void RemoveActiveSource(const int& vertex_id) {
        if (!IsSourceActive(vertex_id)) {
            return;
        }
        active_source_mask_(vertex_id) = 0;
        active_source_ids_.erase(
            remove(active_source_ids_.begin(), active_source_ids_.end(), vertex_id),
            active_source_ids_.end());
    }

    void ToggleActiveSource(const int& vertex_id) {
        if (IsSourceActive(vertex_id)) {
            RemoveActiveSource(vertex_id);
        } else {
            AddActiveSource(vertex_id);
        }
    }

    void ApplyAffinitySourceColors() {
        if (!use_structural_affinity_mode_ || joint_point_selected_) {
            return;
        }
        C_points_cage_ = MatrixXd::Ones(num_vertices_, 3);
        for (int i = 0; i < (int)active_source_ids_.size(); ++i) {
            C_points_cage_.row(active_source_ids_[i]) << 1.0, 0.5, 0.0;
        }
    }

    void EnsureDragSources(const int& fallback_source) {
        drag_source_ids_ = active_source_ids_;
        if (drag_source_ids_.empty() && fallback_source >= 0 && fallback_source < num_vertices_) {
            drag_source_ids_.push_back(fallback_source);
        }
    }

    void SetDepthReferenceFromPoint(const Vector3d& world_point) {
        Vector4d homo = world_point.homogeneous();
        Matrix4d view_matrix = (viewer->core().view).cast<double>();
        Vector4d view_coordinates_homo = view_matrix * homo;
        z_selected_point_ = view_coordinates_homo.hnormalized()(2);
    }

public:
    DeformCageViewerPlugin() {}
    DeformCageViewerPlugin(const MatrixXd& V,
                           const MatrixXd& C,
                           const MatrixXi& F,
                           const int& viewerMeshDataId,
                           const int& viewerCageDataId,
                           const MeanValueCoordController& mVCoord_ctrlr,
                           const StructuralAffinityController& structural_affinity_ctrlr,
                           const double affinity_neighbor_alpha) {
        V_cage_ = V;
        V_cage_deformed_ = V;
        F_cage_ = F;
        viewer_data_cage_id_ = viewerCageDataId;
        viewer_data_mesh_id_ = viewerMeshDataId;
        num_vertices_ = V.rows();
        num_faces_ = F.rows();
        C_points_cage_ = C;
        is_selecting_point_ = false;
        is_moving_point_ = false;
        joint_point_selected_ = false;
        is_selecting_joint_point_ = false;
        current_selected_point_id_ = -1;
        current_joint_selected_id_ = -1;
        z_selected_point_ = 0.;

        mVCoord_controller_ = mVCoord_ctrlr;
        structural_affinity_controller_ = structural_affinity_ctrlr;
        affinity_neighbor_alpha_ = min(1.0, max(0.0, affinity_neighbor_alpha));
        use_structural_affinity_mode_ = false;

        visited_points_ = VectorXi::Zero(num_vertices_);
        active_source_mask_ = VectorXi::Zero(num_vertices_);
        drag_base_cage_ = V_cage_deformed_;

        for (int i = 0; i < num_vertices_; i++) {
            neighbors_.push_back(vector<int>());
        }
        for (int f = 0; f < num_faces_; f++) {
            int current_ind, neighbor_ind;
            for (int i = 0; i < 3; i++) {
                current_ind = F_cage_(f, i);
                for (int j = 0; j < 3; j++) {
                    if (j != i) {
                        neighbor_ind = F_cage_(f, j);
                        neighbors_[current_ind].push_back(neighbor_ind);
                    }
                }
            }
        }
    }

    void ResetVCage(const MatrixXd& V, const MatrixXd& colors) {
        V_cage_ = V;
        V_cage_deformed_ = V;
        is_selecting_point_ = false;
        is_moving_point_ = false;
        is_selecting_joint_point_ = false;
        joint_point_selected_ = false;
        C_points_cage_ = colors;
        points_to_move_.clear();
        visited_points_ = VectorXi::Zero(num_vertices_);
        drag_base_cage_ = V_cage_deformed_;
        drag_source_ids_.clear();
        ClearActiveSources();
    }

    void SetUseStructuralAffinity(const bool enabled) {
        use_structural_affinity_mode_ = enabled;
        if (!enabled) {
            ClearActiveSources();
            if (!joint_point_selected_) {
                C_points_cage_ = MatrixXd::Ones(num_vertices_, 3);
            }
        } else {
            ApplyAffinitySourceColors();
        }
        SyncViewerCage();
    }

    int GetNumActiveSources() const {
        return (int)active_source_ids_.size();
    }

    bool mouse_down(int button, int modifier) {
        int face_id;
        Eigen::Vector3f barycentric_coord;
        // Ray casting
        double x = viewer->current_mouse_x;
        double y = viewer->core().viewport(3) - viewer->current_mouse_y;

        if (button == GLFW_MOUSE_BUTTON_LEFT && (is_selecting_point_ || is_selecting_joint_point_)) {
            if (igl::unproject_onto_mesh(Vector2f(x, y),
                                         viewer->core().view,
                                         viewer->core().proj,
                                         viewer->core().viewport,
                                         V_cage_deformed_,
                                         F_cage_,
                                         face_id,
                                         barycentric_coord)) {
                double max_bary = DBL_MIN;
                int i_max = -1;
                for (int i = 0; i < 3; i++) {
                    if (barycentric_coord(i) > max_bary) {
                        max_bary = barycentric_coord(i);
                        i_max = i;
                    }
                }

                const int v_selected = F_cage_(face_id, i_max);
                if (is_selecting_joint_point_) {
                    if (joint_point_selected_) {
                        joint_point_selected_ = false;
                        C_points_cage_.row(current_joint_selected_id_) << 1, 1, 1;
                    } else {
                        joint_point_ = V_cage_deformed_.row(v_selected);
                        C_points_cage_.row(v_selected) << 0, 0, 1;
                        current_joint_selected_id_ = v_selected;
                        joint_point_selected_ = true;
                        ClearActiveSources();
                    }
                    SyncViewerCage();
                } else {
                    if (joint_point_selected_) {
                        points_to_move_.clear();
                        visited_points_ = VectorXi::Zero(num_vertices_);
                        double max_dist = (Vector3d(V_cage_deformed_.row(v_selected)) - joint_point_).norm();
                        AddPointsToMove(v_selected, max_dist, v_selected, 0);
                        V_cage_ = V_cage_deformed_;
                        for (int i = 0; i < (int)points_to_move_.size(); i++) {
                            int ind = points_to_move_[i];
                            C_points_cage_.row(ind) << 0, 1, 0;
                        }
                        SyncViewerCage();
                    } else {
                        const bool shift_pressed = (modifier & GLFW_MOD_SHIFT) != 0;
                        current_selected_point_id_ = v_selected;

                        if (use_structural_affinity_mode_) {
                            if (shift_pressed) {
                                ToggleActiveSource(v_selected);
                                ApplyAffinitySourceColors();
                                SyncViewerCage();
                                return true;
                            }

                            if (!IsSourceActive(v_selected)) {
                                ClearActiveSources();
                                AddActiveSource(v_selected);
                            }

                            ApplyAffinitySourceColors();
                            drag_base_cage_ = V_cage_deformed_;
                            EnsureDragSources(v_selected);
                        } else {
                            C_points_cage_.row(current_selected_point_id_) << 1, 0, 0;
                            drag_source_ids_.clear();
                        }
                        SyncViewerCage();
                    }

                    Vector2d mouse_pos = Vector2d(x, y);
                    SetDepthReferenceFromPoint(Vector3d(V_cage_deformed_.row(v_selected)));
                    initial_mouse_world_pos_ = MouseToWorld(mouse_pos);
                    is_moving_point_ = true;
                }
                return true;
            }
        }
        return false;
    }

    void AddPointsToMove(const int& current_ind,
                         const double& max_dist,
                         const int& start_ind,
                         const double& sum_dists) {
        double dist = (V_cage_deformed_.row(current_ind) - V_cage_deformed_.row(start_ind)).norm();
        if (dist < max_dist && sum_dists < MAX_EDGE_SUM_DISTANCES_RATIO_ * max_dist) {
            points_to_move_.push_back(current_ind);
            visited_points_(current_ind) = 1;
            vector<int> current_neighbors = neighbors_[current_ind];
            int neighbor_ind;
            for (int i = 0; i < (int)current_neighbors.size(); i++) {
                neighbor_ind = current_neighbors[i];
                if (visited_points_(neighbor_ind) == 0) {
                    double new_sum_dists =
                        sum_dists +
                        (V_cage_deformed_.row(current_ind) - V_cage_deformed_.row(neighbor_ind)).norm();
                    AddPointsToMove(neighbor_ind, max_dist, start_ind, new_sum_dists);
                }
            }
        }
    }

    Vector3d MouseToWorld(const Vector2d& mouse_pos) {
        Vector4d homo;
        // Ray casting
        Vector3d s, dir;
        igl::unproject_ray(mouse_pos, viewer->core().view, viewer->core().proj, viewer->core().viewport, s, dir);
        Vector4d dir_4d = (viewer->core().view).cast<double>() * dir.homogeneous();
        Vector3d dir_view = dir_4d.hnormalized();  // position in view world
        Vector3d current_view_position = dir_view * z_selected_point_ / dir_view(2);

        homo = ((viewer->core().view).cast<double>()).inverse() * current_view_position.homogeneous();
        return homo.hnormalized();
    }

    bool mouse_move(int button, int modifier) {
        if (is_moving_point_) {
            double x = viewer->current_mouse_x;
            double y = viewer->core().viewport(3) - viewer->current_mouse_y;
            Vector2d mouse_pos(x, y);
            Vector3d current_mouse_world_position = MouseToWorld(mouse_pos);

            if (joint_point_selected_) {
                Vector3d initial_v = (initial_mouse_world_pos_ - joint_point_).normalized();
                Vector3d new_v = (current_mouse_world_position - joint_point_).normalized();
                Vector3d new_w = (new_v.cross(initial_v)).normalized();
                Vector3d new_u = (new_v.cross(new_w)).normalized();
                Vector3d initial_u = (initial_v.cross(new_w)).normalized();

                Vector3d new_position, initial_position;
                for (int i = 0; i < (int)points_to_move_.size(); i++) {
                    int ind = points_to_move_[i];
                    initial_position = V_cage_.row(ind);
                    new_position = current_mouse_world_position +
                                   (initial_position - initial_mouse_world_pos_).dot(initial_v) * new_v +
                                   (initial_position - initial_mouse_world_pos_).dot(initial_u) * new_u +
                                   (initial_position - initial_mouse_world_pos_).dot(new_w) * new_w;
                    V_cage_deformed_.row(ind) = new_position;
                }
            } else if (use_structural_affinity_mode_ && structural_affinity_controller_.IsReady()) {
                if (drag_source_ids_.empty()) {
                    EnsureDragSources(current_selected_point_id_);
                }
                Vector3d drag_delta = current_mouse_world_position - initial_mouse_world_pos_;
                vector<Vector3d> source_deltas(drag_source_ids_.size(), drag_delta);
                structural_affinity_controller_.ApplyDeltasWithPinnedSources(
                    drag_source_ids_,
                    source_deltas,
                    drag_base_cage_,
                    V_cage_deformed_,
                    affinity_neighbor_alpha_);
            } else {
                V_cage_deformed_.row(current_selected_point_id_) = current_mouse_world_position;
            }

            SyncViewerCage();
            mVCoord_controller_.SetDeformedCage(V_cage_deformed_);
            viewer->data_list[viewer_data_mesh_id_].set_vertices(mVCoord_controller_.MVInterpolate());
            return true;
        }
        return false;
    }

    bool mouse_up(int button, int modifier) {
        if (button == GLFW_MOUSE_BUTTON_LEFT && is_selecting_point_) {
            if (joint_point_selected_) {
                for (int i = 0; i < (int)points_to_move_.size(); i++) {
                    int ind = points_to_move_[i];
                    C_points_cage_.row(ind) << 1, 1, 1;
                }
            } else {
                if (use_structural_affinity_mode_) {
                    ApplyAffinitySourceColors();
                } else if (current_selected_point_id_ >= 0 &&
                           current_selected_point_id_ < num_vertices_) {
                    C_points_cage_.row(current_selected_point_id_) << 1, 1, 1;
                }
            }
            SyncViewerCage();
            is_moving_point_ = false;
            return true;
        }
        return false;
    }

    bool key_down(int key, int modifiers) {
        if (key == 'S') {
            is_selecting_point_ = true;
            return true;
        }
        if (key == 'J') {
            is_selecting_joint_point_ = true;
            if (use_structural_affinity_mode_) {
                ClearActiveSources();
                if (!joint_point_selected_) {
                    C_points_cage_ = MatrixXd::Ones(num_vertices_, 3);
                    SyncViewerCage();
                }
            }
            return true;
        }
        if (key == 'P') {
            viewer->data_list[viewer_data_cage_id_].clear();
            viewer->data_list[viewer_data_cage_id_].set_mesh(V_cage_deformed_, F_cage_);
            viewer->data_list[viewer_data_cage_id_].show_faces = false;
            viewer->data_list[viewer_data_cage_id_].show_lines = true;
            return true;
        }
        return false;
    }

    bool key_up(int key, int modifiers) {
        if (key == 'S') {
            is_selecting_point_ = false;
            return true;
        }
        if (key == 'J') {
            is_selecting_joint_point_ = false;
            return true;
        }
        return false;
    }
};

#endif /* DeformCageViewerPlugin_hpp */
