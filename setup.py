from setuptools import setup

setup(
    name="ald",
    version="0.1",
    description="Active Langevin dynamics.",
    url="https://github.com/zpeng2/ald",
    author="Zhiwei Peng",
    author_email="zhiweipeng1@gmail.com",
    python_requires=">=3.4",
    install_requires=["pycuda", "numpy", "jinja2", "h5py"],
    license="MIT",
    packages=["ald", "ald.rtp", "ald.core"],
    zip_safe=False,
)
