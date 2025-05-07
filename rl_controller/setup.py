from setuptools import setup
from glob import glob
import os

package_name = 'rl_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.yml*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pa',
    maintainer_email='lupusorina@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_controller = rl_controller.rl_controller:main'
        ],
    },
)
