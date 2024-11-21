from setuptools import setup, find_packages

setup(
    name='quosc',
    version='0.2.0',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'matplotlib',
        'joblib',
        'trimesh',
        'scikit-image',
        'scikit-learn',
        'pandas',
        'shapely',
        'rtree', 
        'seaborn'
    ],
    description='Simulating quantum oscillations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mihirm2305/quosc',
    author='Mihir Manium',
    # author_email='your.email@example.com',
    license='MIT',
)
