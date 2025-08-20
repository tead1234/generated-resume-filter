from setuptools import setup, find_packages

setup(
    name="generated-resume-filter",
    version="0.1.0",
    description="AI로 만들어진 자소서를 필터링하는 오픈소스",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.21.0",
        "tqdm>=4.64.0",
    ],
)