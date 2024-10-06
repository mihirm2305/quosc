from setuptools import setup, find_packages

setup(
    name='quosc',
    version='0.1.0',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    description='Simulating quantum oscillations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mihirm2305/quosc',
    author='Your Name',
    author_email='your.email@example.com',
    license='MIT',
)
