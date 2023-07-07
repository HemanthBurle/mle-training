from setuptools import setup

setup(
    name="custom_package",
    version="0.3",
    description="Median housing value prediction",
    author="HemanthBurle",
    author_email="hemanth.burle@tigeranalytics.com",
    packages=["src"],
    install_requires=["numpy", "pandas", "sklearn", "scipy", "six", "argparse"],
)
