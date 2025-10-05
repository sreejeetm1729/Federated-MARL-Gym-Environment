from setuptools import setup, find_packages

setup(
    name="marl_env",
    version="0.1.0",
    description="Federated MARL grid environments compatible with Gymnasium",
    packages=find_packages(),
    install_requires=["gymnasium>=0.29", "numpy>=1.23", "pygame>=2.1", "pillow>=9.0"],
)
