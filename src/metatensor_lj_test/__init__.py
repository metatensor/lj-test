from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
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
    Get a :py:class:`MetatensorAtomisticModel` corresponding to a shifted Lennard-Jones
    potential for the given ``atomic_type``.

    The energy for this model is a sum over all pairs within the cutoff of:

    .. math::

        4 \\epsilon \\left(
            \\left( \\frac{\\sigma}{r} \\right)^12 - \\left( \\frac{\\sigma}{r}
            \\right)^6
        \\right) + \\text{shift}

    Where :math:`r` is the distance between atoms in the pair, and ``shift`` is a
    constant that ensure that the energy given by the above formula goes to 0 at the
    cutoff.

    :param atomic_type: atomic type to which sigma/epsilon correspond
    :param cutoff: spherical cutoff of the model
    :param epsilon: epsilon parameter of Lennard-Jones
    :param sigma: sigma parameter of Lennard-Jones
    :param length_unit: unit used by ``sigma``
    :param energy_unit: unit used by ``epsilon``
    :param with_extension: should the model use the custom TorchScript extension?
    """

    if with_extension:
        from .extension import LennardJonesExtension

        model = LennardJonesExtension(cutoff=cutoff, epsilon=epsilon, sigma=sigma)
    else:
        from .pure import LennardJonesPurePyTorch

        model = LennardJonesPurePyTorch(cutoff=cutoff, epsilon=epsilon, sigma=sigma)

    capabilities = ModelCapabilities(
        length_unit=length_unit,
        interaction_range=cutoff,
        atomic_types=[atomic_type],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit=energy_unit,
                per_atom=True,
                explicit_gradients=[],
            ),
            "energy_ensemble": ModelOutput(
                quantity="energy",
                unit=energy_unit,
                per_atom=True,
                explicit_gradients=[],
            ), # This is a dummy ensemble for testing, returning the same energy many times
            "energy_uncertainty": ModelOutput(
                quantity="energy",
                unit=energy_unit,
                per_atom=False,
                explicit_gradients=[],
            ),
            "non_conservative_forces": ModelOutput(
                quantity="force",
                unit="eV/Angstrom",
                per_atom=True,
                explicit_gradients=[],
            ),
            "non_conservative_stress": ModelOutput(
                quantity="pressure",
                unit="eV/Angstrom^3",
                per_atom=False,
                explicit_gradients=[],
            ),
        },
        supported_devices=["cpu", "cuda", "mps"],
        dtype="float64",
    )

    metadata = ModelMetadata(
        name="Test Lennard-Jones" + (" (with extension)" if with_extension else ""),
        description="""Minimal shifted Lennard-Jones potential, to be used when testing
the integration of metatensor atomistic models with various simulation engines.""",
        authors=["Guillaume Fraux <guillaume.fraux@epfl.ch>"],
        references={
            "model": [
                "https://github.com/luthaf/metatensor-lj-test",
            ],
            "implementation": [
                "https://github.com/lab-cosmo/metatensor",
            ],
        },
    )

    model.eval()
    return MetatensorAtomisticModel(model, metadata, capabilities)
