from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


setup(
    name="pinn-error-bounds",
    version="2.0.0",
    author="Akhilesh Yadav",
    description="Validated a posteriori error bounds for soft-BC PINN solutions of coercive elliptic PDEs",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "."},
    packages=find_packages(
        where=".",
        exclude=[
            "legacy",
            "legacy.*",
            "results",
            "figures",
            "paper",
            "pinn_env",
            "pinn_env.*",
        ],
    ),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
    ],
)
