from setuptools import setup, find_packages

setup(
    name='project2',
    version='1.0',
    author='Vijay Kumar Reddy Gade',
    author_email='vi.gade@ufl.edu',
    packages=find_packages(exclude=('tests', 'docs', 'resources')),
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
