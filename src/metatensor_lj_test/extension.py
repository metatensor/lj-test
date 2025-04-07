import os
import sys
from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelOutput, NeighborListOptions, System

_HERE = os.path.dirname(__file__)


def _lib_path():
    if sys.platform.startswith("darwin"):
        path = os.path.join(_HERE, "lib", "libmetatensor_lj_test.dylib")
    elif sys.platform.startswith("linux"):
        path = os.path.join(_HERE, "lib", "libmetatensor_lj_test.so")
    elif sys.platform.startswith("win"):
        path = os.path.join(_HERE, "bin", "metatensor_lj_test.dll")
    else:
        raise ImportError("Unknown platform. Please edit this file")

    if os.path.isfile(path):
        return path

    raise ImportError("Could not find metatensor_torch shared library at " + path)


class LennardJonesExtension(torch.nn.Module):
    """
    Implementation of Lennard-Jones potential using a custom TorchScript extension,
    following the metatensor atomistic models interface.
    """

    def __init__(self, cutoff, epsilon, sigma):
        super().__init__()
        self._nl_options = NeighborListOptions(cutoff=cutoff, full_list=False, strict=True)

        self._epsilon = epsilon
        self._sigma = sigma

        # shift the energy to 0 at the cutoff
        self._shift = 4 * epsilon * ((sigma / cutoff) ** 12 - (sigma / cutoff) ** 6)

        # load the C++ operators and custom classes
        torch.ops.load_library(_lib_path())

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        if (
            "energy" not in outputs
            and "energy_ensemble" not in outputs
            and "energy_uncertainty" not in outputs
            and "energy_uncertainty" not in outputs
            and "non_conservative_forces" not in outputs
            and "non_conservative_stress" not in outputs
        ):
            return {}
        
        if "energy_ensemble" in outputs and "energy" not in outputs:
            raise ValueError("energy_ensemble cannot be calculated without energy")
        
        if "energy_uncertainty" in outputs and "energy" not in outputs:
            raise ValueError("energy_uncertainty cannot be calculated without energy")
        
        if "non_conservative_forces" in outputs:
            raise ValueError("the model with extensions does not support non-conservative forces")
        
        if "non_conservative_stress" in outputs:
            raise ValueError("the model with extensions does not support non-conservative stress")

        per_atoms = outputs["energy"].per_atom

        all_energies = []
        # Initialize device so we can access it outside of the for loop
        device = torch.device("cpu")
        for system_i, system in enumerate(systems):
            device = system.device

            neighbors = system.get_neighbor_list(self._nl_options)
            pairs = neighbors.samples.view(["first_atom", "second_atom"]).values
            distances = neighbors.values.reshape(-1, 3)

            # call the custom operator
            energy = torch.ops.metatensor_lj_test.lennard_jones(
                pairs=pairs,
                distances=distances,
                n_atoms=len(system),
                epsilon=self._epsilon,
                sigma=self._sigma,
                shift=self._shift,
            )

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
        return_dict = {
            "energy": TensorMap(
                Labels("_", torch.tensor([[0]], device=device)), [block]
            ),
        }

        if "energy_ensemble" in outputs:
            # returns the same energy for all ensemble members
            n_ensemble_members = 16
            return_dict["energy_ensemble"] = TensorMap(
                return_dict["energy"].keys,
                [
                    TensorBlock(
                        values=block.values.reshape(1, -1).repeat(1, n_ensemble_members),
                        samples=block.samples,
                        components=block.components,
                        properties=Labels(["energy"], torch.arange(n_ensemble_members, device=device, dtype=torch.int).reshape(-1, 1)),
                    )
                ]
            )

        if "energy_uncertainty" in outputs:
            # returns an uncertainty of 0.001 * n_atoms^2 (note that the natural
            # scaling would be with sqrt(n_atoms) or n_atoms); this is useful in tests
            # so we can artificially increase the uncertainty with the number of atoms
            n_atoms = torch.tensor([len(system) for system in systems], device=device)
            n_atoms = n_atoms.reshape(-1, 1).to(dtype=systems[0].positions.dtype)
            energy_uncertainty = 0.001 * n_atoms * n_atoms
            return_dict["energy_uncertainty"] = TensorMap(
                return_dict["energy"].keys,
                [
                    TensorBlock(
                        values=energy_uncertainty,
                        samples=Labels(["system"], torch.arange(len(systems), device=device).reshape(-1, 1)),
                        components=block.components,
                        properties=block.properties,
                    )
                ]
            )

        return return_dict

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self._nl_options]
