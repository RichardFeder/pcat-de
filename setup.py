from setuptools import setup

setup(name="pcat-de",
      url="https://github.com/RichardFeder/pcat-de",
      version='0.0.1',
      author="Richard M. Feder",
      author_email="rfederst@caltech.edu",
      packages=["pcat-de"],
      license="MIT",
      description=("Probabilistic cataloging in the presence of diffuse signals."),
      package_data={
          "":
          ["LICENSE"]
      },
      package_dir={'': 'pcat-de/'},
      include_package_data=True,
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Natural Language :: English", "Programming Language :: Python",
          "Operating System :: OS Independent",
          "Intended Audience :: Science/Research"
      ])