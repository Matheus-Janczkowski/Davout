import os

import shutil

import subprocess

import sys

import traceback

# Defines a function to verify if the setup.py file is in the current
# directory

def check_setup_file(package_name):

    if not os.path.exists("setup.py"):

        # Prints the errors

        print("Error: 'setup.py' not found!")

        print("Current directory: "+str(os.getcwd()))

        print("Please run this script from the root of your package.")

        # Exits the system

        sys.exit(1)

    else:

        print("File 'setup.py' was found. Installation of '"+str(
        package_name)+"' will be resumed")

# Defines a function to clean previous installations

def clean_previous_builds(package_name):

    # Sets a list of directories to remove

    directories_to_remove = ['build', 'dist', 'egg-info']
    
    print("\nCleaning previous builds")

    # Iterates through the elements of the current directory

    for item in os.listdir('.'):

        # Verifies if the item is one of the directories to be removed

        if item.endswith('.egg-info') or (item in directories_to_remove):

            # Tries to remove

            try:

                if os.path.isdir(item):

                    shutil.rmtree(item)

                else:

                    os.remove(item)

                print("\nRemoved: '"+str(item)+"'")

            except Exception as e:

                print("\nAn error occurred while removing: '"+str(item)+
                "'\n"+str(e)+"")

                traceback.print_exc()

# Defines a function to build the package

def build_package(package_name):

    print("\nBuilding package '"+str(package_name)+"'")

    # Uses a subprocess to execute the setup.py script

    result = subprocess.run([sys.executable, "setup.py", "bdist_wheel", 
    "sdist"])
    
    if result.returncode != 0:

        print("\nBuild of the package '"+str(package_name)+"' failed!")

        sys.exit(1)

# Defines a function to install the package using pip

def install_package(package_name):

    print("\nInstalling package...")

    # Runs pip to install the build

    result = subprocess.run([sys.executable, "-m", "pip", "install", 
    "."])

    if result.returncode != 0:

        print("\nInstallation of the package '"+str(package_name)+"' f"+
        "ailed!")

        sys.exit(1)

    else:
        
        print("\nSuccessfully installed the package '"+str(package_name
        )+"' ")

# Defines a function to verify the installation by trying to import it

def verify_installation(package_name):

    package_name = str(package_name)

    print("\nVerifying import of '"+package_name+"'")
    
    # Executes command to import the package
    
    command = [sys.executable, "-c", "import "+package_name+"; print('"+
    "   Success! "+package_name+" version:', "+package_name+".__versio"+
    "n__ if hasattr("+package_name+", '__version__') else 'installed')"]
    
    result = subprocess.run(command, cwd=os.path.expanduser("~"), 
    capture_output=False)
    
    if result.returncode == 0:

        print("\n'"+package_name+"' installed and verified successfull"+
        "y!")

    else:

        print("\nImport of the package '"+package_name+"' failed. The "+
        "package installed, but 'import "+package_name+"' threw an err"+
        "or.")

        sys.exit(1)

if __name__ == "__main__":

    # Ensure we are in the script's directory

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Sets the name of the package to be installed

    package_name = "Davout"

    check_setup_file(package_name)
    
    clean_previous_builds(package_name)

    build_package(package_name)

    install_package(package_name)

    verify_installation(package_name)