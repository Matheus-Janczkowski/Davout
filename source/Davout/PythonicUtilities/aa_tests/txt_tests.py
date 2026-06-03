# Routine to test the txt reader 

from ..file_handling_tools import txt_toList

from ..path_tools import get_parent_path_of_file

read_file = txt_toList("combinations_simulation_Lp4_global.txt", 
parent_path=get_parent_path_of_file())

for row in read_file:

    print(row)

    print("")