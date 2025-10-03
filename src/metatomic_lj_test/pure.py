from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap, multiply
from metatomic.torch import ModelOutput, NeighborListOptions, System


class LennardJonesPurePyTorch(torch.nn.Module):
    """
    Pure PyTorch implementation of Lennard-Jones potential, following metatomic models
    interface.
    """

    def __init__(self, cutoff, epsilon, sigma):
        super().__init__()
        self._nl_options = NeighborListOptions(
            cutoff=cutoff, full_list=False, strict=True
        )

        self._epsilon = epsilon
        self._sigma = sigma

        # shift the energy to 0 at the cutoff
        self._shift = 4 * epsilon * ((sigma / cutoff) ** 12 - (sigma / cutoff) ** 6)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        if (
            "energy" not in outputs
            and "energy/doubled" not in outputs
            and "energy_ensemble" not in outputs
            and "energy_ensemble/doubled" not in outputs
            and "energy_uncertainty" not in outputs
            and "energy_uncertainty/doubled" not in outputs
            and "non_conservative_forces" not in outputs
            and "non_conservative_forces/doubled" not in outputs
            and "non_conservative_stress" not in outputs
            and "non_conservative_stress/doubled" not in outputs
        ):
            return {}

        if "energy_ensemble" in outputs and "energy" not in outputs:
            raise ValueError("energy_ensemble cannot be calculated without energy")

        all_energies = []
        all_energies_per_atom = []
        all_non_conservative_forces = []
        all_non_conservative_stress = []

        # Initialize device/dtype so we can access it outside of the for loop
        device = torch.device("cpu")
        dtype = torch.float32
        for system_i, system in enumerate(systems):
            device = system.device
            dtype = system.positions.dtype

            neighbors = system.get_neighbor_list(self._nl_options)
            all_i = neighbors.samples.column("first_atom").to(torch.long)
            all_j = neighbors.samples.column("second_atom").to(torch.long)

            energy = torch.zeros(len(system), dtype=dtype, device=device)

            distances = neighbors.values.reshape(-1, 3)

            sigma_r_6 = (self._sigma / torch.linalg.vector_norm(distances, dim=1)) ** 6
            sigma_r_12 = sigma_r_6 * sigma_r_6
            e = 4.0 * self._epsilon * (sigma_r_12 - sigma_r_6) - self._shift

            # We only compute each pair once (full_list=False in self._nl_options),
            # and assign half of the energy to each atom
            energy = energy.index_add(0, all_i, e, alpha=0.5)
            energy = energy.index_add(0, all_j, e, alpha=0.5)

            if selected_atoms is not None:
                current_system_mask = selected_atoms.column("system") == system_i
                current_atoms = selected_atoms.column("atom")
                current_atoms = current_atoms[current_system_mask].to(torch.long)
                energy = energy[current_atoms]

            all_energies_per_atom.append(energy)
            all_energies.append(energy.sum(0, keepdim=True))

            if (
                "non_conservative_forces" in outputs
                or "non_conservative_forces/doubled" in outputs
            ):
                # we fill the non-conservative forces as the negative gradient of the potential
                # with respect to the positions, plus a random term
                forces = torch.zeros(len(system), 3, device=device, dtype=dtype)
                forces_per_pair = (
                    12.0
                    * self._epsilon
                    * (sigma_r_6 - 2.0 * sigma_r_12)
                    / torch.linalg.vector_norm(distances, dim=1) ** 2
                )
                forces_per_pair = forces_per_pair.unsqueeze(1) * distances
                forces = forces.index_add(0, all_i, forces_per_pair)
                forces = forces.index_add(0, all_j, -forces_per_pair)
                forces = forces + 0.1 * torch.randn_like(forces) * torch.mean(
                    torch.abs(forces)
                )

                if selected_atoms is not None:
                    current_system_mask = selected_atoms.column("system") == system_i
                    current_atoms = selected_atoms.column("atom")
                    current_atoms = current_atoms[current_system_mask].to(torch.long)
                    forces = forces[current_atoms]

                all_non_conservative_forces.append(forces)

            if (
                "non_conservative_stress" in outputs
                or "non_conservative_stress/doubled" in outputs
            ):
                # we fill the non-conservative stress with random numbers
                stress = torch.randn((3, 3), device=device, dtype=dtype)
                all_non_conservative_stress.append(stress)

        energy_values = torch.vstack(all_energies).reshape(-1, 1)
        energies_per_atom_values = torch.vstack(all_energies_per_atom).reshape(-1, 1)

        if (
            "non_conservative_forces" in outputs
            or "non_conservative_forces/doubled" in outputs
        ):
            nc_forces_values = torch.cat(all_non_conservative_forces).reshape(-1, 3, 1)
        else:
            nc_forces_values = torch.empty((0, 0))

        if selected_atoms is None:
            samples_list: List[List[int]] = []
            for s, system in enumerate(systems):
                for a in range(len(system)):
                    samples_list.append([s, a])

            # randomly shuffle the samples to make sure the different engines handle
            # out of order samples
            indexes = torch.randperm(len(samples_list))
            if ("energy" in outputs and outputs["energy"].per_atom) or (
                "energy/doubled" in outputs and outputs["energy/doubled"].per_atom
            ):
                energies_per_atom_values = energies_per_atom_values[indexes]

            if (
                "non_conservative_forces" in outputs
                or "non_conservative_forces/doubled" in outputs
            ):
                nc_forces_values = nc_forces_values[indexes]

            per_atom_samples = Labels(
                ["system", "atom"], torch.tensor(samples_list, device=device)[indexes]
            )
        else:
            per_atom_samples = selected_atoms

        per_system_samples = Labels(
            ["system"], torch.arange(len(systems), device=device).reshape(-1, 1)
        )
        single_key = Labels("_", torch.tensor([[0]], device=device))

        if ("energy" in outputs and outputs["energy"].per_atom) or (
            "energy/doubled" in outputs and outputs["energy/doubled"].per_atom
        ):
            energy_block = TensorBlock(
                values=energies_per_atom_values,
                samples=per_atom_samples,
                components=torch.jit.annotate(List[Labels], []),
                properties=Labels(["energy"], torch.tensor([[0]], device=device)),
            )
        else:
            energy_block = TensorBlock(
                values=energy_values,
                samples=per_system_samples,
                components=torch.jit.annotate(List[Labels], []),
                properties=Labels(["energy"], torch.tensor([[0]], device=device)),
            )

        results: Dict[str, TensorMap] = {}
        if "energy" in outputs or "energy/doubled" in outputs:
            result = TensorMap(single_key, [energy_block])
            if "energy" in outputs:
                results["energy"] = result
            if "energy/doubled" in outputs:
                results["energy/doubled"] = multiply(result, 2.0)

        if "energy_ensemble" in outputs or "energy_ensemble/doubled" in outputs:
            # returns the same energy for all ensemble members
            n_ensemble_members = 16
            ensembled_properties = Labels(
                ["energy"],
                torch.arange(n_ensemble_members, device=device).reshape(-1, 1),
            )

            for variant in ["energy_ensemble", "energy_ensemble/doubled"]:
                if variant not in outputs:
                    continue
                
                if outputs[variant].per_atom:
                    ensemble_block = TensorBlock(
                        values=energies_per_atom_values.repeat(1, n_ensemble_members),
                        samples=per_atom_samples,
                        components=[],
                        properties=ensembled_properties,
                    )
                else:
                    ensemble_block = TensorBlock(
                        values=energy_values.repeat(1, n_ensemble_members),
                        samples=per_system_samples,
                        components=[],
                        properties=ensembled_properties,
                    )

                result= TensorMap(single_key, [ensemble_block])
                if variant == "energy_ensemble/doubled":
                    result = multiply(result,2.0)
                results[variant] = result

        if "energy_uncertainty" in outputs or "energy_uncertainty/doubled" in outputs:
            # returns an uncertainty of `0.001 * n_atoms^2` (note that the natural
            # scaling would be `sqrt(n_atoms)` or `n_atoms`); this is useful in tests so
            # we can artificially increase the uncertainty with the number of atoms
            n_atoms = torch.tensor([len(system) for system in systems], device=device)
            energy_uncertainty = 0.001 * n_atoms**2
            energy_uncertainty = energy_uncertainty.to(dtype=dtype).to(device=device)

            for variant in ["energy_uncertainty", "energy_uncertainty/doubled"]:
                if variant not in outputs:
                    continue

                if outputs[variant].per_atom:
                    selected_systems = per_atom_samples.column("system")
                    energy_uncertainty = torch.vstack(
                        [
                            energy_uncertainty[s].repeat(
                                len(torch.where(selected_systems == s)[0]), 1
                            )
                            for s in torch.unique(selected_systems)
                        ]
                    )

                    uncertainty_block = TensorBlock(
                        values=energy_uncertainty,
                        samples=per_atom_samples,
                        components=[],
                        properties=energy_block.properties,
                    )
                else:
                    uncertainty_block = TensorBlock(
                        values=energy_uncertainty.reshape(-1,1),
                        samples=per_system_samples,
                        components=[],
                        properties=energy_block.properties,
                    )

                result = TensorMap(single_key, [uncertainty_block])
                if variant == "energy_uncertainty/doubled":
                    result = multiply(result, 2.0)
                results[variant] = result

        if (
            "non_conservative_forces" in outputs
            or "non_conservative_forces/doubled" in outputs
        ):
            result = TensorMap(
                keys=Labels("_", torch.tensor([[0]], device=device)),
                blocks=[
                    TensorBlock(
                        values=nc_forces_values,
                        samples=per_atom_samples,
                        components=[
                            Labels(
                                ["xyz"],
                                torch.arange(3, device=device).reshape(-1, 1),
                            )
                        ],
                        properties=Labels(
                            ["non_conservative_forces"],
                            torch.tensor([[0]], device=device),
                        ),
                    )
                ],
            )
            if "non_conservative_forces" in outputs:
                results["non_conservative_forces"] = result
            if "non_conservative_forces/doubled" in outputs:
                results["non_conservative_forces/doubled"] = multiply(result, 2.0)

        if (
            "non_conservative_stress" in outputs
            or "non_conservative_stress/doubled" in outputs
        ):
            result = TensorMap(
                keys=Labels("_", torch.tensor([[0]], device=device)),
                blocks=[
                    TensorBlock(
                        values=torch.cat(all_non_conservative_stress).reshape(
                            -1, 3, 3, 1
                        ),
                        samples=Labels(
                            ["system"],
                            torch.arange(len(systems), device=device).reshape(-1, 1),
                        ),
                        components=[
                            Labels(
                                ["xyz_1"],
                                torch.arange(3, device=device).reshape(-1, 1),
                            ),
                            Labels(
                                ["xyz_2"],
                                torch.arange(3, device=device).reshape(-1, 1),
                            ),
                        ],
                        properties=Labels(
                            ["non_conservative_stress"],
                            torch.tensor([[0]], device=device),
                        ),
                    )
                ],
            )
            if "non_conservative_stress" in outputs:
                results["non_conservative_stress"] = result
            if "non_conservative_stress/doubled" in outputs:
                results["non_conservative_stress/doubled"] = multiply(result, 2.0)

        return results

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self._nl_options]
