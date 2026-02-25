//
//  main.cpp
//  3DA_project_CageBasedDef_bin
//
//  Created by Benjamin Barral on 05/02/2019.
//

#include <stdio.h>
#define DIM_X 0
#define DIM_Y 1
#define DIM_Z 2


#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <iostream>
#include <igl/readPLY.h>
#include <igl/writePLY.h>
#include <igl/writeOBJ.h>
#include <igl/file_exists.h>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <iterator>
#include <limits>
#include "MeanValueCoordController.hpp"
#include "MeshProcessor.hpp"
#include <random>
#include "CageGenerator.hpp"
#include <igl/opengl/glfw/Viewer.h>
#include "DeformCageViewerPlugin.hpp"
#include "StructuralAffinityController.hpp"


using namespace std;
using namespace Eigen;
using namespace igl::opengl::glfw;


string mesh_file_name = "../../left_gripper.obj";
string lower_cage_file_name = "";
string upper_cage_file_name = "";

bool compute_automatic_cage = true;
float sparseness_cage = 0.5; // For automatic cage generation : CHANGE THIS for a sparser or denser cage : the larger the parameter the denser the cage
float upper_cage_sparseness_ratio = 0.6; // Upper cage is generated from lower cage with a lower sparseness.
bool use_upper_convex_fallback = false;
int upper_convex_target_vertices = 8;

static bool ParseBoolArg(const string& text, bool& value) {
    string lowered = text;
    transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return (char) tolower(c);
    });
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
        value = true;
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
        value = false;
        return true;
    }
    return false;
}

static bool ParseDoubleArg(const string& text, double& value) {
    try {
        size_t parsed_chars = 0;
        value = stod(text, &parsed_chars);
        return parsed_chars == text.size() && isfinite(value);
    } catch (...) {
        return false;
    }
}

static bool ParseIntArg(const string& text, int& value) {
    try {
        size_t parsed_chars = 0;
        long parsed = stol(text, &parsed_chars);
        if (parsed_chars != text.size()) {
            return false;
        }
        if (parsed < numeric_limits<int>::min() || parsed > numeric_limits<int>::max()) {
            return false;
        }
        value = (int) parsed;
        return true;
    } catch (...) {
        return false;
    }
}

static double Clamp01(const double value) {
    return min(1.0, max(0.0, value));
}

static void PrintUsage(const char* exe_name) {
    cout << "Usage: " << exe_name << " [options]" << endl;
    cout << "  --mesh <path>                    Base mesh file path (.obj/.ply/.off)" << endl;
    cout << "  --generate-cage <bool>           true/false, 1/0 (default: true)" << endl;
    cout << "  --lower-sparsity <float>         Sparsity for lower cage generation (default: 0.5)" << endl;
    cout << "  --upper-sparsity-ratio <float>   Upper sparsity = lower_sparsity * ratio (default: 0.6)" << endl;
    cout << "  --lower-cage <path>              Lower cage path when --generate-cage false" << endl;
    cout << "  --upper-cage <path>              Upper cage path when --generate-cage false" << endl;
    cout << "  --upper-convex-fallback <bool>   If true, skip upper generation and use convex-hull-like upper cage around lower cage (default: false)" << endl;
    cout << "  --upper-convex-vertices <int>    Target vertex count for convex-hull-like upper cage when enabled (default: 8)" << endl;
    cout << "  --affinity-epsilon <float>       Epsilon threshold for affinity matrix (default: 1e-4)" << endl;
    cout << "  --affinity-target-density <f>    Target density in [0,1], overrides --affinity-epsilon by searching epsilon" << endl;
    cout << "  --affinity-target-sparsity <f>   Deprecated alias of --affinity-target-density" << endl;
    cout << "  --affinity-alpha <float>         Scale [0,1] for non-source propagated cage motion (default: 1.0)" << endl;
    cout << "  --help                           Show this message" << endl;
}

static double FindAffinityEpsilonForTargetDensity(const MatrixXd& lower_to_upper_weights,
                                                  const double target_density,
                                                  const int max_iterations,
                                                  const double tolerance,
                                                  double& out_density,
                                                  int& out_nnz) {
    const double clamped_target = Clamp01(target_density);
    StructuralAffinityController tmp_controller;

    double lower = 0.0;
    double upper = 1.0;
    double best_epsilon = 0.0;
    double best_error = numeric_limits<double>::infinity();
    out_density = 0.0;
    out_nnz = 0;

    auto Evaluate = [&](const double epsilon) -> double {
        tmp_controller.BuildFromWeights(lower_to_upper_weights, epsilon);
        const double density = tmp_controller.GetDensity();
        const double error = abs(density - clamped_target);
        if (error < best_error) {
            best_error = error;
            best_epsilon = epsilon;
            out_density = density;
            out_nnz = tmp_controller.GetNumNonZeros();
        }
        return density;
    };

    Evaluate(lower);
    Evaluate(upper);
    for (int iter = 0; iter < max_iterations; ++iter) {
        const double epsilon = 0.5 * (lower + upper);
        const double density = Evaluate(epsilon);
        if (abs(density - clamped_target) <= tolerance) {
            break;
        }
        if (density > clamped_target) {
            lower = epsilon;
        } else {
            upper = epsilon;
        }
    }

    return best_epsilon;
}

static void BuildConvexBoundingCage(const MatrixXd& V_source,
                                    MatrixXd& V_convex,
                                    MatrixXi& F_convex,
                                    const double padding_ratio,
                                    const int target_vertices) {
    if (V_source.rows() == 0) {
        V_convex.resize(0, 3);
        F_convex.resize(0, 3);
        return;
    }

    Vector3d min_point = V_source.colwise().minCoeff();
    Vector3d max_point = V_source.colwise().maxCoeff();
    Vector3d extents = max_point - min_point;
    const double diag = extents.norm();
    const double pad = max(1e-6, padding_ratio * max(diag, 1e-6));
    min_point.array() -= pad;
    max_point.array() += pad;

    const int desired_vertices = max(8, target_vertices);
    if (desired_vertices <= 8) {
        V_convex = MatrixXd::Zero(8, 3);
        F_convex = MatrixXi::Zero(12, 3);

        // Vertices
        V_convex.row(0) << min_point(0), min_point(1), min_point(2);
        V_convex.row(1) << max_point(0), min_point(1), min_point(2);
        V_convex.row(2) << max_point(0), min_point(1), max_point(2);
        V_convex.row(3) << min_point(0), min_point(1), max_point(2);
        V_convex.row(4) << min_point(0), max_point(1), min_point(2);
        V_convex.row(5) << max_point(0), max_point(1), min_point(2);
        V_convex.row(6) << max_point(0), max_point(1), max_point(2);
        V_convex.row(7) << min_point(0), max_point(1), max_point(2);

        // Triangles (convex cube)
        F_convex.row(0) << 0, 1, 2;
        F_convex.row(1) << 0, 2, 3;
        F_convex.row(2) << 1, 5, 6;
        F_convex.row(3) << 1, 6, 2;
        F_convex.row(4) << 5, 4, 7;
        F_convex.row(5) << 5, 7, 6;
        F_convex.row(6) << 4, 0, 3;
        F_convex.row(7) << 4, 3, 7;
        F_convex.row(8) << 4, 5, 1;
        F_convex.row(9) << 4, 1, 0;
        F_convex.row(10) << 3, 2, 6;
        F_convex.row(11) << 3, 6, 7;
        return;
    }

    // Build a convex-hull-like support cage with near-target vertex count.
    int best_lon = 3;
    int best_lat = max(1, desired_vertices - 2);
    int best_actual = 2 + best_lat * best_lon;
    int best_error = abs(best_actual - desired_vertices);
    for (int lon = 3; lon <= desired_vertices - 2; ++lon) {
        int lat = max(1, (int) llround((double) (desired_vertices - 2) / lon));
        int actual = 2 + lat * lon;
        int error = abs(actual - desired_vertices);
        if (error < best_error || (error == best_error && actual < best_actual)) {
            best_lon = lon;
            best_lat = lat;
            best_actual = actual;
            best_error = error;
        }
    }

    const int lon_steps = best_lon;
    const int lat_steps = best_lat;
    const int num_vertices = 2 + lat_steps * lon_steps;
    const int num_faces = 2 * lat_steps * lon_steps;

    V_convex = MatrixXd::Zero(num_vertices, 3);
    F_convex = MatrixXi::Zero(num_faces, 3);

    const Vector3d center = 0.5 * (min_point + max_point);
    const double pi = acos(-1.0);

    auto SupportPoint = [&](const Vector3d& direction) -> Vector3d {
        Vector3d dir = direction;
        const double n = dir.norm();
        if (n <= 1e-12) {
            return center;
        }
        dir /= n;

        double support = -numeric_limits<double>::infinity();
        for (int vi = 0; vi < V_source.rows(); ++vi) {
            const Vector3d offset = Vector3d(V_source.row(vi)) - center;
            support = max(support, offset.dot(dir));
        }
        if (!isfinite(support)) {
            support = 0.;
        }
        const Vector3d point = center + (support + pad) * dir;
        return point;
    };

    V_convex.row(0) = SupportPoint(Vector3d::UnitZ()).transpose();
    for (int lat = 1; lat <= lat_steps; ++lat) {
        const double phi = pi * (double) lat / (double) (lat_steps + 1);
        const double sin_phi = sin(phi);
        const double cos_phi = cos(phi);
        for (int lon = 0; lon < lon_steps; ++lon) {
            const double theta = 2.0 * pi * (double) lon / (double) lon_steps;
            const double x = sin_phi * cos(theta);
            const double y = sin_phi * sin(theta);
            const double z = cos_phi;
            const Vector3d point = SupportPoint(Vector3d(x, y, z));
            const int index = 1 + (lat - 1) * lon_steps + lon;
            V_convex.row(index) = point.transpose();
        }
    }
    const int south_idx = num_vertices - 1;
    V_convex.row(south_idx) = SupportPoint(-Vector3d::UnitZ()).transpose();

    int face_idx = 0;
    // North cap
    for (int lon = 0; lon < lon_steps; ++lon) {
        const int curr = 1 + lon;
        const int next = 1 + (lon + 1) % lon_steps;
        F_convex.row(face_idx++) << 0, curr, next;
    }
    // Middle rings
    for (int lat = 0; lat < lat_steps - 1; ++lat) {
        for (int lon = 0; lon < lon_steps; ++lon) {
            const int top_curr = 1 + lat * lon_steps + lon;
            const int top_next = 1 + lat * lon_steps + (lon + 1) % lon_steps;
            const int bot_curr = 1 + (lat + 1) * lon_steps + lon;
            const int bot_next = 1 + (lat + 1) * lon_steps + (lon + 1) % lon_steps;
            F_convex.row(face_idx++) << top_curr, bot_curr, top_next;
            F_convex.row(face_idx++) << top_next, bot_curr, bot_next;
        }
    }
    // South cap
    const int last_ring_base = 1 + (lat_steps - 1) * lon_steps;
    for (int lon = 0; lon < lon_steps; ++lon) {
        const int curr = last_ring_base + lon;
        const int next = last_ring_base + (lon + 1) % lon_steps;
        F_convex.row(face_idx++) << south_idx, next, curr;
    }
}

int main(int argc, char *argv[])
{
    double affinity_epsilon = 1e-4;
    double affinity_target_density = -1.0;
    double affinity_neighbor_alpha = 1.0;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--help") {
            PrintUsage(argv[0]);
            return 0;
        } else if (arg == "--mesh" && i + 1 < argc) {
            mesh_file_name = argv[++i];
        } else if (arg == "--lower-cage" && i + 1 < argc) {
            lower_cage_file_name = argv[++i];
        } else if (arg == "--upper-cage" && i + 1 < argc) {
            upper_cage_file_name = argv[++i];
        } else if (arg == "--upper-convex-fallback" && i + 1 < argc) {
            bool parsed = false;
            if (!ParseBoolArg(argv[++i], parsed)) {
                cout << "Invalid value for --upper-convex-fallback" << endl;
                return 1;
            }
            use_upper_convex_fallback = parsed;
        } else if (arg == "--upper-convex-vertices" && i + 1 < argc) {
            int parsed = 0;
            if (!ParseIntArg(argv[++i], parsed)) {
                cout << "Invalid value for --upper-convex-vertices" << endl;
                return 1;
            }
            upper_convex_target_vertices = max(8, parsed);
        } else if (arg == "--generate-cage" && i + 1 < argc) {
            bool parsed = false;
            if (!ParseBoolArg(argv[++i], parsed)) {
                cout << "Invalid value for --generate-cage" << endl;
                return 1;
            }
            compute_automatic_cage = parsed;
        } else if (arg == "--lower-sparsity" && i + 1 < argc) {
            double parsed = 0.;
            if (!ParseDoubleArg(argv[++i], parsed)) {
                cout << "Invalid value for --lower-sparsity" << endl;
                return 1;
            }
            sparseness_cage = (float) max(0.001, parsed);
        } else if (arg == "--upper-sparsity-ratio" && i + 1 < argc) {
            double parsed = 0.;
            if (!ParseDoubleArg(argv[++i], parsed)) {
                cout << "Invalid value for --upper-sparsity-ratio" << endl;
                return 1;
            }
            upper_cage_sparseness_ratio = (float) max(0.001, parsed);
        } else if (arg == "--affinity-epsilon" && i + 1 < argc) {
            double parsed = 0.;
            if (!ParseDoubleArg(argv[++i], parsed)) {
                cout << "Invalid value for --affinity-epsilon" << endl;
                return 1;
            }
            affinity_epsilon = max(0.0, parsed);
        } else if ((arg == "--affinity-target-density" || arg == "--affinity-target-sparsity") && i + 1 < argc) {
            double parsed = 0.;
            if (!ParseDoubleArg(argv[++i], parsed)) {
                cout << "Invalid value for " << arg << endl;
                return 1;
            }
            affinity_target_density = Clamp01(parsed);
        } else if (arg == "--affinity-alpha" && i + 1 < argc) {
            double parsed = 0.;
            if (!ParseDoubleArg(argv[++i], parsed)) {
                cout << "Invalid value for --affinity-alpha" << endl;
                return 1;
            }
            affinity_neighbor_alpha = min(1.0, max(0.0, parsed));
        } else {
            cout << "Unknown or incomplete argument: " << arg << endl;
            PrintUsage(argv[0]);
            return 1;
        }
    }

    srand(time(NULL));
    clock_t start;

    cout << "Configuration:" << endl;
    cout << "  mesh = " << mesh_file_name << endl;
    cout << "  generate_cage = " << (compute_automatic_cage ? "true" : "false") << endl;
    cout << "  lower_sparsity = " << sparseness_cage << endl;
    cout << "  upper_sparsity_ratio = " << upper_cage_sparseness_ratio << endl;
    cout << "  upper_convex_fallback = " << (use_upper_convex_fallback ? "true" : "false") << endl;
    cout << "  upper_convex_vertices = " << upper_convex_target_vertices << endl;
    cout << "  affinity_epsilon = " << affinity_epsilon << endl;
    if (affinity_target_density >= 0.0) {
        cout << "  affinity_target_density = " << affinity_target_density
             << " (epsilon search enabled)" << endl;
    } else {
        cout << "  affinity_target_density = disabled" << endl;
    }
    cout << "  affinity_alpha = " << affinity_neighbor_alpha << endl;
    
    // LOAD MESHES
    MatrixXd V_mesh, V_cage, V_cage_upper;
    MatrixXi F_mesh, F_cage, F_cage_upper;
    if (mesh_file_name.find("obj") != string::npos) igl::readOBJ(mesh_file_name,V_mesh,F_mesh);
    else if (mesh_file_name.find("ply") != string::npos) igl::readPLY(mesh_file_name,V_mesh,F_mesh);
    else if (mesh_file_name.find("off") != string::npos) igl::readOFF(mesh_file_name,V_mesh,F_mesh);
    else {
        cout << "Mesh file not recognized " << endl;
        return 1;
    }
    
    // Init the viewer
    Viewer viewer;
    viewer.core().is_animating = true;
    viewer.append_mesh();
    viewer.append_mesh();
    viewer.append_mesh();
    float point_size = 18.;
    
    // Generate automatic cage
    MatrixXd V_cage_automatic, V_cage_automatic_smooth;
    float lambda_smooth_implicit = .6;
    int num_iterations_smoothing = 2;
    CageGenerator cage_generator(V_mesh, F_mesh, num_iterations_smoothing, lambda_smooth_implicit, sparseness_cage);
    if (compute_automatic_cage) {
        // Generate cage
        MatrixXd bb_vertices;
        MatrixXi bb_faces;
        vector<MatrixXd> feat_voxels_vertices;
        vector<MatrixXi> feat_voxels_faces;
        start = clock();
        cage_generator.ComputeCage(bb_vertices, bb_faces, feat_voxels_vertices, feat_voxels_faces);
        cout << "Time to compute cage : " << (clock() - start) / (double) CLOCKS_PER_SEC << endl;
        
        V_cage_automatic = cage_generator.GetCage();
        V_cage_automatic_smooth = cage_generator.GetSmoothCage();
        //MatrixXd V_cage_automatic = cage_generator.GetSmoothCage();
        MatrixXi F_cage_automatic = cage_generator.GetCageFaces();
        // Show automatically generated cage
        V_cage = V_cage_automatic_smooth;
        F_cage = F_cage_automatic;
        
        // Save mesh in PLY file
        igl::writePLY("../../GeneratedCage.ply", V_cage_automatic_smooth, F_cage_automatic);
    }
    else {
        if (lower_cage_file_name.find("obj") != string::npos) igl::readOBJ(lower_cage_file_name,V_cage,F_cage);
        else if (lower_cage_file_name.find("ply") != string::npos) igl::readPLY(lower_cage_file_name,V_cage,F_cage);
        else if (lower_cage_file_name.find("off") != string::npos) igl::readOFF(lower_cage_file_name,V_cage,F_cage);
        else {
            cout << "Lower cage file not recognized " << endl;
            return 1;
        }
    }

    if (use_upper_convex_fallback) {
        cout << "Upper convex option enabled. Skipping upper generation and building a convex-hull-like upper cage around lower cage." << endl;
        BuildConvexBoundingCage(V_cage, V_cage_upper, F_cage_upper, 0.05, upper_convex_target_vertices);
        cout << "Built convex upper cage with " << V_cage_upper.rows()
             << " vertices and " << F_cage_upper.rows() << " faces." << endl;
        igl::writePLY("../../GeneratedUpperCageConvex.ply", V_cage_upper, F_cage_upper);
    } else if (compute_automatic_cage) {
        // Build upper cage from the generated lower cage.
        const float upper_sparseness = max(0.05f, sparseness_cage * upper_cage_sparseness_ratio);
        CageGenerator upper_cage_generator(V_cage, F_cage, num_iterations_smoothing, lambda_smooth_implicit, upper_sparseness);
        MatrixXd bb_vertices_upper;
        MatrixXi bb_faces_upper;
        vector<MatrixXd> feat_voxels_vertices_upper;
        vector<MatrixXi> feat_voxels_faces_upper;

        start = clock();
        upper_cage_generator.ComputeCage(bb_vertices_upper,
                                         bb_faces_upper,
                                         feat_voxels_vertices_upper,
                                         feat_voxels_faces_upper);
        cout << "Time to compute upper cage from lower cage : "
             << (clock() - start) / (double) CLOCKS_PER_SEC << endl;

        V_cage_upper = upper_cage_generator.GetSmoothCage();
        F_cage_upper = upper_cage_generator.GetCageFaces();

        if (V_cage_upper.rows() == 0 || F_cage_upper.rows() == 0 || !V_cage_upper.allFinite()) {
            cout << "Upper cage generation failed and convex option is disabled." << endl;
            return 1;
        } else {
            igl::writePLY("../../GeneratedUpperCage.ply", V_cage_upper, F_cage_upper);
        }
    } else {
        if (upper_cage_file_name.find("obj") != string::npos) igl::readOBJ(upper_cage_file_name, V_cage_upper, F_cage_upper);
        else if (upper_cage_file_name.find("ply") != string::npos) igl::readPLY(upper_cage_file_name, V_cage_upper, F_cage_upper);
        else if (upper_cage_file_name.find("off") != string::npos) igl::readOFF(upper_cage_file_name, V_cage_upper, F_cage_upper);
        else {
            cout << "Upper cage file not recognized " << endl;
            return 1;
        }
        if (V_cage_upper.rows() == 0 || F_cage_upper.rows() == 0) {
            cout << "Upper cage load failed or empty." << endl;
            return 1;
        }
    }
    
    // Get barycenter and extreme points of mesh
    MeshProcessor cage_processor(V_cage,F_cage);
    Vector3d cage_barycenter = cage_processor.GetBarycenter();
    MeshProcessor mesh_processor(V_mesh,F_mesh);
    Vector3d mesh_barycenter = mesh_processor.GetBarycenter();
    int dim = DIM_X;
    int max_vertY_ind = cage_processor.GetMaximumVertexIndex(dim);
    double bb_size = mesh_processor.GetBoundingBoxSize();
    
    // Print out number of triangles
    int num_faces_mesh = F_mesh.rows();
    int num_vertices_cage = V_cage.rows();
    int num_faces_cage = F_cage.rows();
    int num_vertices_upper_cage = V_cage_upper.rows();
    int num_faces_upper_cage = F_cage_upper.rows();
    int num_vertices_mesh = V_mesh.rows();
    cout << "Lower cage : " << num_faces_cage << " triangles, " << num_vertices_cage << " vertices." << endl;
    cout << "Upper cage : " << num_faces_upper_cage << " triangles, " << num_vertices_upper_cage << " vertices." << endl;
    cout << "Mesh : " << num_faces_mesh << " triangles, " << num_vertices_mesh << " vertices." << endl;
    
    
    // Visualize axes
    MatrixXd axes_points1 = MatrixXd::Zero(3, 3);
    MatrixXd axes_points2 = MatrixXd::Zero(3, 3);
    axes_points2.coeffRef(0, 0) = 1;
    axes_points2.coeffRef(1,1) = 1;
    axes_points2.coeffRef(2,2) = 1;
    MatrixXd axes_colors = axes_points2;
    axes_points2 *= 0.25;
    double offset_z = 1.3;
    for (int i = 0; i<3; i++){
        axes_points2.coeffRef(i, 2) += offset_z;
        axes_points1.coeffRef(i, 2) += offset_z;
    }
    viewer.data_list[0].add_edges(axes_points1, axes_points2, axes_colors);
    
    viewer.data_list[1].set_mesh(V_mesh, F_mesh);
    viewer.data_list[1].show_lines = false;
    MatrixXd cage_points_colors = MatrixXd::Ones(num_vertices_cage,3);
    viewer.data_list[2].set_mesh(V_cage, F_cage);
    viewer.data_list[2].show_faces = false;
    viewer.data_list[2].show_lines = true;
    viewer.data_list[2].add_points(V_cage, cage_points_colors);
    viewer.data_list[2].point_size = point_size;

    // Upper cage is visualized as wireframe only.
    viewer.data_list[3].set_mesh(V_cage_upper, F_cage_upper);
    viewer.data_list[3].show_faces = false;
    viewer.data_list[3].show_lines = true;
            
    // Mean Value Coordinates for mesh deformation from lower cage
    MeanValueCoordController mVCoord_controller(V_mesh, V_cage, F_mesh, F_cage, bb_size);
    start = clock();
    mVCoord_controller.ComputeMVWeights();
    cout << "Time to compute weights (mesh->lower cage) : "
         << (clock() - start) / (double) CLOCKS_PER_SEC << endl;
    start = clock();
    MatrixXd V_mesh_deformed = V_mesh;
    V_mesh_deformed = mVCoord_controller.MVInterpolate();
    float timeInterp = (clock() - start) / (double) CLOCKS_PER_SEC;
    float fps = 1. / timeInterp;
    cout << "Time to interpolate : " << timeInterp << endl;
    cout << "fps = " << fps << endl;

    MatrixXd V_cage_deformed = V_cage;
    mVCoord_controller.SetDeformedCage(V_cage_deformed);

    // Mean Value Coordinates for lower cage vertices in upper cage weight-space
    MeanValueCoordController lower_to_upper_controller(V_cage, V_cage_upper, F_cage, F_cage_upper, bb_size);
    start = clock();
    lower_to_upper_controller.ComputeMVWeights();
    cout << "Time to compute weights (lower->upper cage) : "
         << (clock() - start) / (double) CLOCKS_PER_SEC << endl;

    StructuralAffinityController structural_affinity_controller;
    const double target_density_search_tolerance = 1e-3;
    const int target_density_search_max_iterations = 24;
    double effective_affinity_epsilon = affinity_epsilon;
    double searched_density = 0.0;
    int searched_nnz = 0;

    if (affinity_target_density >= 0.0) {
        start = clock();
        effective_affinity_epsilon = FindAffinityEpsilonForTargetDensity(
            lower_to_upper_controller.GetWeights(),
            affinity_target_density,
            target_density_search_max_iterations,
            target_density_search_tolerance,
            searched_density,
            searched_nnz);
        cout << "Affinity epsilon search (target density = " << affinity_target_density
             << ") found epsilon = " << effective_affinity_epsilon
             << ", density = " << searched_density
             << ", nnz = " << searched_nnz
             << " in " << (clock() - start) / (double) CLOCKS_PER_SEC << " s" << endl;
    }

    start = clock();
    structural_affinity_controller.BuildFromWeights(
        lower_to_upper_controller.GetWeights(),
        effective_affinity_epsilon);
    cout << "Time to precompute structural affinity : "
         << (clock() - start) / (double) CLOCKS_PER_SEC << endl;
    cout << "Affinity matrix non-zeros: " << structural_affinity_controller.GetNumNonZeros()
         << " (density " << structural_affinity_controller.GetDensity() << ")" << endl;
    const double effective_affinity_density = structural_affinity_controller.GetDensity();

    // Attach a plugin to handle deformation interaction
    int cage_data_id = 2, mesh_data_id = 1;
    DeformCageViewerPlugin def_cage_plugin(V_cage,
                                           cage_points_colors,
                                           F_cage,
                                           mesh_data_id,
                                           cage_data_id,
                                           mVCoord_controller,
                                           structural_affinity_controller,
                                           affinity_neighbor_alpha);
    viewer.plugins.push_back(&def_cage_plugin);
    
    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    def_cage_plugin.init(&viewer);
    viewer.plugins.push_back(&menu);
    
    // Wave deformation
    float wave_time_start;
    bool wave_isActive = false;
    float wave_duration = .75;
    int wave_index_point;
    
    
    // Deform cage
    float cage_resize_ratioZ = 1.,  cage_resize_ratioY = 1., cage_resize_ratioX = 1.;
    bool use_structural_affinity_drag = false;
    int smooth_cage_slider = 1;
    
    // UI instructions
    cout << endl;
    cout << "CAGE INTERACTION : " << endl;
    cout << "J + click on a face : select/unselect a joint point (appears in blue)" << endl;
    cout << "S + click on a face : " << endl;
    cout << "   If joint point selected : select group of points at distance lesser than joint point from the click (appear in green) by rotating around the joint point" << endl;
    cout << "   If no joint point selected : move one point (appears in red) " << endl;
    cout << "S + SHIFT + click : toggle multiple structural-affinity sources when affinity mode is enabled" << endl;
    cout << "P : Hide points" << endl;
    
    menu.callback_draw_custom_window = [&]()
    {
        bool update_cage = false;
        bool update_mesh = false;
        // Define next window position + size
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(800, 100), ImGuiCond_FirstUseEver);
        ImGui::Begin(
                     "MyProperties", nullptr,
                     ImGuiWindowFlags_NoSavedSettings
                     );
        ImGui::End();

        if (ImGui::Checkbox("Structural Affinity Drag", &use_structural_affinity_drag)) {
            def_cage_plugin.SetUseStructuralAffinity(use_structural_affinity_drag);
        }
        ImGui::Text("Affinity epsilon (effective) = %.1e", effective_affinity_epsilon);
        if (affinity_target_density >= 0.0) {
            ImGui::Text("Affinity target density = %.6f", affinity_target_density);
        }
        ImGui::Text("Affinity alpha = %.3f", affinity_neighbor_alpha);
        ImGui::Text("Affinity nnz = %d (density = %.6f)",
                    structural_affinity_controller.GetNumNonZeros(),
                    structural_affinity_controller.GetDensity());
        ImGui::Text("Affinity density = %.6f", effective_affinity_density);
        ImGui::Text("Active sources = %d (Shift+click to toggle)",
                    def_cage_plugin.GetNumActiveSources());
        
        
        if (ImGui::Button("Show interpolated")){
            viewer.data_list[1].clear();
            viewer.data_list[1].set_mesh(V_mesh_deformed, F_mesh);
            viewer.data_list[2].clear();
            viewer.data_list[2].set_mesh(V_cage_deformed, F_cage);
            viewer.data_list[2].show_faces = false;
            viewer.data_list[2].show_lines = true;
            viewer.data_list[2].set_points(V_cage_deformed, cage_points_colors);
            viewer.data_list[2].point_size = point_size;
        }
        if (ImGui::Button("Show original")){
            viewer.data_list[1].clear();
            viewer.data_list[1].set_mesh(V_mesh, F_mesh);
            viewer.data_list[2].clear();
            viewer.data_list[2].set_mesh(V_cage, F_cage);
            viewer.data_list[2].show_faces = false;
            viewer.data_list[2].show_lines = true;
            viewer.data_list[2].set_points(V_cage, cage_points_colors);
            viewer.data_list[2].point_size = point_size;
        }
        if (ImGui::SliderFloat("Stretching cage on Z axis", &cage_resize_ratioZ, 0, 2)){
            for (int i = 0; i < num_vertices_cage; i++){
                V_cage_deformed(i,DIM_Z) = cage_barycenter(DIM_Z) +
                (V_cage(i,DIM_Z) -  cage_barycenter(DIM_Z)) * cage_resize_ratioZ;
            }
            update_mesh = true;
            update_cage = true;
        }
        if (ImGui::SliderFloat("Stretching cage on Y axis", &cage_resize_ratioY, 0, 2)){
            for (int i = 0; i < num_vertices_cage; i++){
                V_cage_deformed(i,DIM_Y) = cage_barycenter(DIM_Y) +
                (V_cage(i,DIM_Y) -  cage_barycenter(DIM_Y)) * cage_resize_ratioY;
            }
            update_mesh = true;
            update_cage = true;
        }
        if (ImGui::SliderFloat("Stretching cage on X axis", &cage_resize_ratioX, 0, 2)){
            for (int i = 0; i < num_vertices_cage; i++){
                V_cage_deformed(i,DIM_X) = cage_barycenter(DIM_X) +
                (V_cage(i,DIM_X) -  cage_barycenter(DIM_X)) * cage_resize_ratioX;
            }
            update_mesh = true;
            update_cage = true;
        }
     
        if (compute_automatic_cage) {
            if (ImGui::SliderInt("Show coarse/smooth cage", &smooth_cage_slider, 0, 1)){
                V_cage_deformed = (smooth_cage_slider == 1) ? V_cage_automatic_smooth : V_cage_automatic;
                update_cage = true;
            }
            ImGui::SliderFloat("lamba smoothing", &lambda_smooth_implicit, 0,2);
            ImGui::SliderInt("Smoothing iterations", &num_iterations_smoothing, 1, 15);
            if (ImGui::Button("Smooth Cage")) {
                cage_generator.SetSmoothingParameters(num_iterations_smoothing, lambda_smooth_implicit);
                cage_generator.SmoothCage();
                V_cage_automatic_smooth = cage_generator.GetSmoothCage();
                V_cage_deformed = V_cage_automatic_smooth;
                update_cage = true;
            }
        }
        
        if (ImGui::Button("Reset")){
            V_cage_deformed = V_cage;
            V_mesh_deformed = V_mesh;
            viewer.data_list[1].set_vertices(V_mesh_deformed);
            update_cage = true;
        }
        
        if(ImGui::Button("Random point wave deformation")){
            wave_index_point = rand() % num_vertices_cage;
            wave_isActive = true;
            wave_time_start = clock();
        }
        ImGui::SliderFloat("Wave duration", &wave_duration, 0.2, 2);
        
        if (wave_isActive){
            
            float t = 2. * (clock() - wave_time_start) /((double) CLOCKS_PER_SEC * wave_duration);
            if (t > 1.){
                wave_isActive = false;
                V_cage_deformed.row(wave_index_point) = V_cage.row(wave_index_point);
            }
            else{
                double ratio = 1. + sin(M_PI * t);
                Vector3d original_vert = V_cage.row(wave_index_point);
                Vector3d deformed_cage_vert =  cage_barycenter +
                ratio * (original_vert - cage_barycenter);
                V_cage_deformed.row(wave_index_point) = deformed_cage_vert;
            }
            update_mesh = true;
            update_cage = true;
        }
        
        if (update_cage){
            viewer.data_list[2].set_vertices(V_cage_deformed);
            viewer.data_list[2].set_points(V_cage_deformed, cage_points_colors);
            viewer.data_list[2].show_faces = false;
            viewer.data_list[2].show_lines = true;
            def_cage_plugin.ResetVCage(V_cage_deformed, cage_points_colors);
            mVCoord_controller.SetDeformedCage(V_cage_deformed);
        }
        if (update_mesh){
            V_mesh_deformed = mVCoord_controller.MVInterpolate();
            viewer.data_list[1].set_vertices(V_mesh_deformed);
        }
        
        
            
    };
    
    // Call GUI
    viewer.launch();
 
}
