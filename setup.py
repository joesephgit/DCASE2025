from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='Tan_SNTLNTU_task1',
    version='0.1.0',
    description='SNTLNTU DCASE 2025 Task1 Inference Package',
    author='Ee-Leng TAN',
    author_email="etanel@ntu.edu.sg",
    packages=find_packages(),  # This auto-discovers the inner folder
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'Tan_SNTLNTU_task1': ["resources/*.wav", 'ckpts/*.ckpt'],
    },
    python_requires='>=3.11',
)
