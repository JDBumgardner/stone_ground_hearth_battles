from setuptools import setup, find_packages  # type: ignore

package_name = "hearthstone"
setup(
    name=package_name,
    packages=find_packages(exclude=('tests',))
)
