"""Installation script for the repo's non-vendored packages."""

from setuptools import find_packages, setup

setup(
    name="vla-world-model-control",
    version="0.1.0",
    description="Simulation and real-robot glue code around vendored LeRobot.",
    packages=find_packages(include=["vla_world_model_control*"]),
    python_requires=">=3.10",
    zip_safe=False,
)
