#include <torch/script.h>

torch::Tensor lennard_jones(
    torch::Tensor pairs,
    torch::Tensor distances,
    int64_t n_atoms,
    double epsilon,
    double sigma,
    double shift
) {
    auto sigma_r = sigma / torch::linalg_vector_norm(distances, 2, /* dim */ 1);
    auto sigma_r_6 = torch::pow(sigma_r, 6);
    auto sigma_r_12 = sigma_r_6 * sigma_r_6;
    auto e = 4.0 * epsilon * (sigma_r_12 - sigma_r_6) - shift;

    auto energies = torch::zeros(n_atoms, torch::TensorOptions().dtype(distances.scalar_type()).device(distances.device()));
    auto all_i = pairs.index({torch::indexing::Slice(), 0});
    auto all_j = pairs.index({torch::indexing::Slice(), 1});

    energies = energies.index_add(0, all_i, e, 0.5);
    energies = energies.index_add(0, all_j, e, 0.5);

    return energies;
}

TORCH_LIBRARY(metatensor_lj_test, m) {
    m.def(
        "lennard_jones(Tensor pairs, Tensor distances, int n_atoms, float epsilon, float sigma, float shift) -> Tensor",
        lennard_jones
    );
}
