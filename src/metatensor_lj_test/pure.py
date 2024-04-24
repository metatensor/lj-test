from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelOutput, NeighborListOptions, System


class LennardJonesPurePyTorch(torch.nn.Module):
    """
    Pure PyTorch implementation of Lennard-Jones potential, following metatensor
    atomistic models interface.
    """

    def __init__(self, cutoff, epsilon, sigma):
        super().__init__()
        self._nl_options = NeighborListOptions(cutoff=cutoff, full_list=False)

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
        if "energy" not in outputs:
            return {}

        per_atoms = outputs["energy"].per_atom

        all_energies = []

        # Initialize device so we can access it outside of the for loop
        device = torch.device("cpu")
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

            if per_atoms:
                all_energies.append(energy)
            else:
                all_energies.append(energy.sum(0, keepdim=True))

        if per_atoms:
            if selected_atoms is None:
                samples_list: List[List[int]] = []
                for s, system in enumerate(systems):
                    for a in range(len(system)):
                        samples_list.append([s, a])

                samples = Labels(
                    ["system", "atom"], torch.tensor(samples_list, device=device)
                )
            else:
                samples = selected_atoms
        else:
            samples = Labels(
                ["system"], torch.arange(len(systems), device=device).reshape(-1, 1)
            )

        block = TensorBlock(
            values=torch.vstack(all_energies).reshape(-1, 1),
            samples=samples,
            components=torch.jit.annotate(List[Labels], []),
            properties=Labels(["energy"], torch.tensor([[0]], device=device)),
        )
        return {
            "energy": TensorMap(
                Labels("_", torch.tensor([[0]], device=device)), [block]
            ),
        }

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self._nl_options]
