from setuptools import find_packages, setup

package_name = 'ai_assist'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', '../aria_context.md']),
        ('share/' + package_name + '/config', ['ai_assist/config.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ashish',
    maintainer_email='ashishramesh2003@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'function_call = ai_assist.function_call:main',
            'detect_aruco = ai_assist.detect_aruco :main',
            'dock = ai_assist.dock:main',
            'speech = ai_assist.speech:main',
            'nav_goal = ai_assist.nav_goal:main'
        ],
    },
)
