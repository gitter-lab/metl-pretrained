from setuptools import setup
setup(name='metl-pretrained',
      version='0.1',
      description='Mutational effect transfer learning',
      url='https://github.com/samgelman/metl-pretrained',
      author='Sam Gelman',
      author_email='sgelman2@wisc.edu',
      license='MIT',
      packages=['main'],
      install_requires=['pytorch', 'numpy', 'scipy', 'biopandas'])
