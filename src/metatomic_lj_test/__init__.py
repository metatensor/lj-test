from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
)


def lennard_jones_model(
    atomic_type,
    cutoff,
    epsilon,
    sigma,
    length_unit,
    energy_unit,
    with_extension,
):
    """
    Get a metatomic model corresponding to a shifted Lennard-Jones potential for the
    given ``atomic_type``.

    The energy for this model is a sum over all pairs within the cutoff of:

    .. math::

        4 \\epsilon \\left(
            \\left( \\frac{\\sigma}{r} \\right)^12 - \\left( \\frac{\\sigma}{r}
            \\right)^6
        \\right) + \\text{shift}

    Where :math:`r` is the distance between atoms in the pair, and ``shift`` is a
    constant that ensure that the energy given by the above formula goes to 0 at the
    cutoff.

    The model also provides a ``doubled`` **variant** where the :math:`\\epsilon`
    parameter is scaled by a factor of 2.

    :param atomic_type: atomic type to which sigma/epsilon correspond
    :param cutoff: spherical cutoff of the model
    :param epsilon: epsilon parameter of Lennard-Jones
    :param sigma: sigma parameter of Lennard-Jones
    :param length_unit: unit used by ``sigma``
    :param energy_unit: unit used by ``epsilon``
    :param with_extension: should the model use the custom TorchScript extension?
    """

    outputs = {
        "energy": ModelOutput(
            description="LJ energy",
            unit=energy_unit,
            sample_kind="atom",
        ),
        "energy/doubled": ModelOutput(
            description="LJ energy, multiplied by 2",
            unit=energy_unit,
            sample_kind="atom",
        ),
    }

    if with_extension:
        from .extension import LennardJonesExtension

        model = LennardJonesExtension(cutoff=cutoff, epsilon=epsilon, sigma=sigma)
    else:
        from .pure import LennardJonesPurePyTorch

        model = LennardJonesPurePyTorch(cutoff=cutoff, epsilon=epsilon, sigma=sigma)

        outputs.update(
            {
                "energy_ensemble": ModelOutput(
                    description="ensemble of energy",
                    unit=energy_unit,
                    sample_kind="atom",
                ),
                "energy_ensemble/doubled": ModelOutput(
                    description="ensemble of energy, multiplied by 2",
                    unit=energy_unit,
                    sample_kind="atom",
                ),
                "energy_uncertainty": ModelOutput(
                    description="Pseudo uncertainty for tests, scales with the system size",
                    unit=energy_unit,
                    sample_kind="atom",
                ),
                "energy_uncertainty/doubled": ModelOutput(
                    description="Pseudo uncertainty for tests, multiplied by 2",
                    unit=energy_unit,
                    sample_kind="atom",
                ),
                "non_conservative_force": ModelOutput(
                    description="Direct force prediction for LJ",
                    unit="eV/Angstrom",
                    sample_kind="atom",
                ),
                "non_conservative_force/doubled": ModelOutput(
                    description="Direct force prediction for LJ, multiplied by 2",
                    unit="eV/Angstrom",
                    sample_kind="atom",
                ),
                "non_conservative_stress": ModelOutput(
                    description="Random stress prediction",
                    unit="eV/Angstrom^3",
                    sample_kind="system",
                ),
                "non_conservative_stress/doubled": ModelOutput(
                    description="Random stress prediction, multiplied by 2",
                    unit="eV/Angstrom^3",
                    sample_kind="atom",
                ),
            }
        )

    capabilities = ModelCapabilities(
        length_unit=length_unit,
        interaction_range=cutoff,
        atomic_types=[atomic_type],
        outputs=outputs,
        supported_devices=["cpu", "cuda", "mps"],
        dtype="float64",
    )

    metadata = ModelMetadata(
        name="Test Lennard-Jones" + (" (with extension)" if with_extension else ""),
        description="""Minimal shifted Lennard-Jones potential, to be used when testing
the integration of metatomic models with various simulation engines.""",
        authors=["Guillaume Fraux <guillaume.fraux@epfl.ch>"],
        references={
            "model": [
                "https://github.com/metatensor/lj-test",
            ],
            "implementation": [
                "https://github.com/metatensor/metatomic",
            ],
        },
    )

    model.eval()
    return AtomisticModel(model, metadata, capabilities)
