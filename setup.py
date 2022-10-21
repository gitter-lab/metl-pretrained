from setuptools import setup
setup(name='metl-pretrained',
      version='0.1',
      description='Mutational effect transfer learning',
      url='https://github.com/samgelman/metl-pretrained',
      author='Sam Gelman',
      author_email='sgelman2@wisc.edu',
      license='MIT',
      packages=['metl'],
      install_requires=['torch>=1.11.0', 'numpy>=1.23.2', 'scipy>=1.9.1', 'biopandas>=0.2.7'])
