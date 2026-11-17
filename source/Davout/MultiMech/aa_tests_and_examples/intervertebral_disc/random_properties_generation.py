# cd /home/grante_maximus/rafael_prado_ic/TCC_rafael_prado/IntervertebralDiscANN/simulations

import random 
from cylindrical_coordinates_material_property import sup_inf_parameters
from Davout.MultiMech.tool_box.numerical_tools import rotation_tensorEulerRodrigues
from Davout.MultiMech.tool_box.numerical_tools import get_axle_from_rotation_tensor
from Davout.PythonicUtilities.file_handling_tools import list_toTxt
from .....Davout.StochasticUtilities.tool_box.sampling_tools import get_random_point_on_elipsoid_surface
from copy import deepcopy
from geometry_disc import disc_geometry
from local_coordinate_system_generation import coordinates_direction
import numpy as np

# Maximum and minimal values of Nicolini's parameters:

# c_HGO: (0.19; 0.2)
# c1_mooney: (0.14, 0.18)
# c2_mooney: (0.03, 0.03)
# k1: (3.68, 2.88)
# k2: (60.54, 61.86)
# k2_c: (-0.2, -0.19)
# k2_r: (-0.04, -0.03)
# k1_c: (-0.1, -0.13)
# k1_r: (-0.04, -0.15)
# gamma: (33.07, 36.73)
# gamma_c: (0.04, 0.07)
# gamma_r: (0.07, 0.03)
# kappa: (0.09, 0.05)
# ROM_flexion_extenion: (-10.0, 10.0)
# ROM_lateral_bending:(-6.0, 6.0)
# ROM_axial_rotation: (-6.0, 6.0)

# exemplo de lista de argumneto: [[[0.8, 0.9], "kappa", function_scalar], [[60.54, 61.86], [-0.2, -0.19], [-0.04, -0.03], "k2", function_field]]
            
def function_scalar(value, field_name, combination_number):
     
    scalar_value_raised = random.uniform(*value)

    return [scalar_value_raised], [scalar_value_raised]

def function_field(value_base, value_percentage_c,
value_percentage_r, parent_path, 
mesh_file_name, mesh_data_class, field_name, combination_number):
    
    base_value_raised = random.uniform(*value_base)

    percentage_c_raised = random.uniform(*value_percentage_c)

    percentage_r_raised = random.uniform(*value_percentage_r)

    full_file_name = sup_inf_parameters(base_value_raised, percentage_c_raised,
    percentage_r_raised, field_name, parent_path, field_name+"_"+str(combination_number), 
    mesh_file_name, mesh_data_class)

    return [full_file_name], [base_value_raised, percentage_c_raised, percentage_r_raised]

def function_dirichlet(translation_x, translation_y, translation_z,
rotation_x, rotation_y, rotation_z, field_name, combination_number):
    
    # The implemetantion of point searching on the surface of N dimension elipsoid was used

    result = get_random_point_on_elipsoid_surface(
    [translation_x, translation_y, translation_z, rotation_x, rotation_y, rotation_z])

    (translation_x_raised, translation_y_raised, translation_z_raised, 
    rotation_x_raised, rotation_y_raised, rotation_z_raised)=result

    print(rotation_x[0]*(180/np.pi), rotation_x[1]*(180/np.pi))

    print(rotation_x_raised*(180/np.pi), rotation_y_raised*(180/np.pi), rotation_z_raised*(180/np.pi))

    print(result)

    # Calculate the rotation tensor from Euler Rodrigues
    # about the x axis

    tensor_r_x = rotation_tensorEulerRodrigues([rotation_x_raised, 0, 0], return_numpy_array=True)

    print(tensor_r_x)

    # Calculate the rotation tenso from Euler Rodrigues
    # about the y axis

    tensor_r_y = rotation_tensorEulerRodrigues([0, rotation_y_raised, 0], return_numpy_array=True)

    print(tensor_r_y)

    # Calculate the rotation z tensor ""tensor_r_z" 

    tensor_r_z = rotation_tensorEulerRodrigues([0, 0, rotation_z_raised], return_numpy_array=True)

    print(tensor_r_z)

    # Multiply the tensors r_z, r_y, and r_x. The convention is r_x
    # and then r_y and r_z (human column motion)

    rzy_tensor = np.matmul(tensor_r_z, tensor_r_y)

    rzyx_tensor = np.matmul(rzy_tensor, tensor_r_x)

    # Calls the function of MultiMech to get the axle vector.
    # That is the "axle_vector"

    axle_vector = get_axle_from_rotation_tensor(rzyx_tensor, return_numpy_array=True)    

    return [[translation_x_raised, translation_y_raised, translation_z_raised],
             (axle_vector*(180/np.pi)).tolist()],[translation_x_raised, translation_y_raised,
             translation_z_raised, axle_vector.tolist()[0], axle_vector.tolist()[1],
             axle_vector.tolist()[2]]

def generate_paramenter_combination(properties, n_combinations, output_file_name, 
input_file_name_ann):

    # Start a table to control de simulation

    ann_input = []

    table = []

    for property in properties:

        # Get the penultimate object of each sublist "property" (name)

        name = property[-2]

        # Get the last object of each sublist "property" (function will
        # be used to describe the property)

        function = property[-1]

        if function == function_dirichlet:

            # The function_dirichlet generate 4 vectors, so, 
            # it is necesary to create 4 columns in the table

            table.extend(["translation_vector", "rotation_vector"])

            ann_input.extend(["translation_x", "translation_y", "translation_z", "rotation_x", "rotation_y", 
                              "rotation_z"])
        
        elif function == function_field:

            # Add just "path" for the name to indicate that is a archive saved
            # with a path predefined

            table.append(name + "_path")

            ann_input.extend([name + "_base_value", name + "_circumferential_percentage", name + "_radial_percentage"])

        elif function == function_scalar:
            
            # Scalar function

            table.append(name)

            ann_input.append(name)

        else:

            raise NameError("function '" + str(function) + "' is not an allowed function")
    
    # To indicate that the simulation has finalized

    table.append("simulation_done")

    # table is a list, to bild all the main big table
    # is necessary to put in other list, In this way
    # it is possible to do a big table with all informations

    properties_input_ann = [ann_input]

    properties_combinations = [table]

    # Iterate in each properties parameters in list properties

    for i in range(n_combinations):

        properties_input_ann.append([])

        properties_combinations.append([])

        for property in properties:

            table_content, ann_content = property[-1](*property[0:(len(property)-1)], i)
           
            properties_combinations[-1].extend(table_content)

            properties_input_ann[-1].extend(ann_content)
        
        # Sentence VI from overleaf activities, inicate if 
        # the simulation was realized. Later on, it is necessary to 
        # verify if the simulations was realized or not, if not, 
        # the FEM simulation get the table row and use the paramenters
        # sorted

        properties_combinations[-1].append(False)

        # Calls the function to transpose the archive from list to .TXT

        list_toTxt(properties_combinations, output_file_name, parent_path="/media/grante_maximus/Dados/data_set_tcc")

        # Calls the function to transpose new ann_list to apply in RNA
        # code as inputs

        list_toTxt(properties_input_ann, input_file_name_ann, parent_path="/media/grante_maximus/Dados/data_set_tcc")
            
    return properties_combinations, properties_input_ann

if __name__=="__main__":

    factor_unit = 0.001

    # |----------------------------------------------------------------------------|
    # |                      PARENT PATH AND MESH FILE NAME PATH                   |
    # |----------------------------------------------------------------------------|

    terminal_number = 4
 
    parent_path = "/media/grante_maximus/Dados/data_set_tcc/material_properties_"+str(terminal_number)

    mesh_file_name = "/media/grante_maximus/Dados/data_set_tcc/meshes_"+str(terminal_number)+"/intervertebral_disc_mesh"


    # |----------------------------------------------------------------------------|
    # |                   GENERATE GEOMETRIC AND MESH OF THE DISC                  |            
    # |----------------------------------------------------------------------------|

    transfinite_nucleus_radial=6

    transfinite_annulus_radial=6

    transfinite_circumferential=16

    transfinite_axial=6

    bias_radial_nucleus = -1.15

    bias_radial_annulus = -1.15

    bias_axial = ["Bump", 0.3]

    x_top_external = np.array(list(reversed([15.380790185431579, 16.38936396301903, 17.397944152962154, 18.448544509763956, 19.289024795205385, 
         20.255578405934166, 21.13807885823426, 21.978559143675696, 22.734992683044126, 23.3653528971252, 
         23.827619619060265, 24.163806603352285, 24.415953253926993, 24.373926674712635, 24.24785976178098, 
         24.121786436493622, 23.86963978591892, 23.49142622241255, 23.15523282576484, 22.692966103829775, 
         22.188679215036046, 21.642365747027988, 20.969985366088252, 20.339625152007173, 19.58319161263875, 
         18.86878465248467, 18.07032453390189, 17.187824081601796, 16.221270470873016, 15.296740233180728, 
         14.24613987637894, 13.237566098791497, 12.27101248806271, 11.136365385187915, 10.001715076135277, 
         8.993138092369982, 7.858490989495188, 6.807890632693397, 5.631216950604248, 4.286449776369089, 
         3.1938260465307984, 1.7650089200447923, 0.0000016030889185003616, -1.7229855470082924, -2.941679395956097, 
         -4.202399824118245, -5.505146831494741, -6.765867259656893, -7.774441037244337, -8.951114719333484, 
         -10.043738449171776, -11.052315432937066, -12.144940765864284, -13.15351614654065, -14.204116503342442, 
         -15.254716860144233, -16.179245494747594, -17.06174915322553, -17.94425441479239, -18.826758073270327, 
         -19.499143263476824, -20.213550223630904, -20.885935413837405, -21.558319000954985, -22.10463086587413, 
         -22.6509443338822, -23.02915950047749, -23.44939964319821, -23.869639785918924, -24.12178483340471, 
         -24.205831579477714, -24.5, -24.37392827780157, -24.205831579477714, -23.91166476204435, -23.44939964319821, 
         -22.777016056080633, -22.06260749283763, -21.2641505804327, -20.297598572792833, -19.373069938189474, 
         -18.406517930549608, -17.439965922909746, -16.47341391526988, -15.42281355846809, -14.246141479467868, 
         -13.195541122666075, -12.144940765864284, -11.010292059900564, -9.959691703098773, -8.740994647973123, 
         -7.64837091813483, -6.429673863009182, -5.589193577567748, -4.706689919089815, -3.7401363083610235, 
         -2.815609276846587, -1.849055666117799, -0.9665520076398693, -0.04202497612542544, 0.9245286346033552, 
         1.7229855470082889, 2.7735859038100834, 3.74013630836102, 4.664666546053304, 5.505146831494738, 
         6.471697236045678, 7.606347545098323, 8.783018021009624, 9.917668330062263, 11.052315432937057, 
         12.18696574198971, 13.237566098791497, 14.414236574702798])))


    y_top_external = np.array(list(reversed([16.17812468893828, 15.699738710632378, 15.264842894580251, 14.5690102524952, 14.047135356182444, 
     13.30781213709465, 12.568489747504803, 11.69869811540055, 10.828906483296297, 9.611197700651577, 
     8.654427403035665, 7.393228872886148, 6.219009837746228, 4.78385397657339, 3.566146023426615, 
     2.391926988286695, 1.0872395401303194, -0.13046841301645884, -1.304687448156379, -2.391926988286693, 
     -3.4791665284170072, -4.6533855635569275, -5.784117339685865, -6.7843740668148165, -7.871613606945131, 
     -8.697916321042525, -9.480730117133056, -10.263540595231827, -11.046354391322359, -11.61171696139507, 
     -12.307552921471876, -12.872915491544582, -13.35130022560357, -13.7861960416557, -14.308073011713303, 
     -14.655987673759952, -15.177864643817555, -15.482290387857347, -15.873697285902605, -16.134634111935533, 
     -16.352082019961593, -16.526041009980798, -16.61301884599452, -16.526041009980798, -16.395570937968458, 
     -16.17812302994239, -15.873697285902605, -15.438801469850482, -15.134375725810697, -14.74296882776543, 
     -14.221091857707822, -13.742707123648835, -13.220833471582988, -12.785937655530864, -12.177082849459538, 
     -11.61171696139507, -10.959373237316878, -10.307029513238687, -9.56770795314678, -8.697916321042525, 
     -7.741146852924549, -6.827862984821678, -5.653645608677637, -4.696877799555546, -3.5661460234266134, 
     -2.4354159062935548, -1.348176366163237, -0.04348891800686161, 1.1307301171330586, 2.3049474932770977, 
     3.6531255184362124, 4.827344553576133, 6.001561929720168, 7.393228872886148, 8.567447908026068, 
     9.741666943165985, 10.785415076795616, 11.655206708899868, 12.481510252495202, 13.30781213709465, 
     14.047135356182444, 14.612499170502058, 15.177864643817564, 15.743229287635113, 16.17812468893828, 
     16.56953034273663, 16.873958160521266, 17.17838514880796, 17.308853976573392, 17.439321974840883, 
     17.52630188459946, 17.613280964860092, 17.48281213709465, 17.308853976573392, 17.26536422906859, 
     17.134895816052133, 17.004426158788757, 16.917447908026066, 16.917447908026066, 16.873958160521266, 
     16.873958160521266, 17.004426158788757, 17.091406068547325, 17.221874066814816, 17.26536422906859, 
     17.439321974840883, 17.439321974840883, 17.439321974840883, 17.52630188459946, 17.52630188459946, 
     17.352343724078192, 17.091406068547325, 16.917447908026066, 16.56953034273663])))

    z_top_external = np.array(list(reversed([14.0 for i in range(len(x_top_external))])))

    # top_internal_curve_set_points

    x_top_internal = np.array(list(reversed([11.052315432937057, 12.060892416702352, 12.733276003819931, 13.489709543188361, 
         14.120069757269427, 14.624356646063156, 15.212693487107721, 15.716983582079301, 
         16.095197145585658, 16.38936396301903, 16.557463867520724, 16.599484034379387, 
         16.557463867520724, 16.43139054223338, 16.26329705008736, 16.053176978727002, 
         15.75900374893795, 15.506863510718937, 15.170670114071221, 14.708406598314006, 
         14.372213201666291, 13.951973058945576, 13.489709543188361, 12.985419448216788, 
         12.313035861099209, 11.808748972305494, 11.220412131260915, 10.7161252424672, 
         10.043738449171776, 9.455404814305048, 8.783018021009624, 8.110634433892045, 
         7.438250846774469, 6.76586725965689, 6.0934836725393176, 5.379073506207384, 
         4.706689919089804, 4.118353078045239, 3.403946117891156, 2.6475125785227256, 
         1.8910822453321465, 1.218695452036723, 0.0000016030889185003616, -1.09262533292722, 
         -1.93310561836865, -2.5634658324497295, -3.4039461178911594, -4.076329705008735, 
         -4.62263996683896, -5.463120252280397, -6.09348046636147, -6.723840680442542, 
         -7.396227473737969, -8.026587687819045, -8.65694790190012, -9.371354862054199, 
         -10.043738449171776, -10.674098663252853, -11.17838875822442, -11.76672399618007, 
         -12.313035861099218, -12.77530097994536, -13.27958786873908, -13.825901336747151, 
         -14.330188225540873, -14.708404995225086, -15.254716860144233, -15.506861907630016, 
         -15.843053701188806, -16.13722212171109, -16.389365566107955, -16.389365566107955, 
         -16.55746226443181, -16.55746226443181, -16.599485637468312, -16.47341391526988, 
         -16.13722212171109, -15.716981978990376, -15.254716860144233, -14.666381622188583, 
         -14.162093130305939, -13.447684567062938, -12.733276003819935, -12.060892416702355, 
         -11.38850882958478, -10.632075290216347, -9.87564175084792, -9.119211417657338, 
         -8.278731132215906, -7.396227473737969, -6.6818173074060425, -5.841337021964609, 
         -5.042880109559679, -4.3284731494056, -3.5720396100371694, -2.8996560229195936, 
         -2.227272435802014, -1.5548888486844348, -0.798455309316008, -0.04202497612542544, 
         0.6723851902064979, 1.4708421026114316, 2.227272435802014, 2.941682602133934, 
         3.57204281621501, 4.328473149405596, 5.0008567365231755, 5.799313648928106, 
         6.597770561333032, 7.480274219810969, 8.278731132215903, 9.035164671584333, 
         9.917668330062263])))
    

    y_top_internal = np.array(list(reversed([12.264062344469142, 11.742187033407411, 11.307291217355285, 10.828906483296297, 
     10.26354142472977, 9.741666943165985, 9.04583181258711, 8.39348808850892, 
     7.6541665284170115, 6.827864643817563, 5.95807301171331, 5.131770297615915, 
     4.305469242514405, 3.4356776104101527, 2.609374896312758, 1.7830738412112481, 
     1.0437506221234543, 0.17395899001920512, -0.5653642290685852, -1.348176366163237, 
     -2.0875029032427896, -2.870311722345681, -3.4791665284170072, -4.088019675492456, 
     -4.696877799555546, -5.17525921562277, -5.697134526684502, -6.088541424729764, 
     -6.52343724078189, -6.827862984821678, -7.132292046853223, -7.480206708899866, 
     -7.784635770931411, -8.002083678957472, -8.263020504990399, -8.52395733102332, 
     -8.741405239049389, -8.915364229068587, -9.13281213709465, -9.393748963127575, 
     -9.524219035139915, -9.611196871153638, -9.611196871153638, -9.611196871153638, 
     -9.480730117133056, -9.35026004512071, -9.219789973108373, -8.915364229068587, 
     -8.784894157056243, -8.48046841301646, -8.263020504990399, -8.045572596964334, 
     -7.784635770931411, -7.480206708899866, -7.175780964860085, -6.7843740668148165, 
     -6.52343724078189, -6.088541424729764, -5.697134526684502, -5.262238710632376, 
     -4.740363399570647, -4.218489747504801, -3.5661460234266134, -2.913802299348422, 
     -2.130988503257889, -1.3916669431659798, -0.5653642290685852, 0.21744790802606317, 
     0.9132822091069954, 1.7395832642085054, 2.609374896312758, 3.39218703340741, 
     4.218489747504801, 5.088281379609054, 5.871093516703706, 6.740885148807958, 
     7.697656275921812, 8.436979495009602, 9.04583181258711, 9.6981755366653, 
     10.30703117223457, 10.698437240781896, 11.307291217355285, 11.69869811540055, 
     12.133593931452676, 12.481510252495202, 12.655468413016461, 12.872916321042528, 
     13.003385563556929, 13.220833471582996, 13.177343724078192, 13.177343724078192, 
     13.220833471582996, 13.220833471582996, 13.09036422906859, 13.003385563556929, 
     12.872916321042528, 12.742447908026069, 12.742447908026069, 12.698958160521265, 
     12.698958160521265, 12.785937655530866, 12.916406068547325, 12.959895816052128, 
     13.046875311061733, 13.133853976573388, 13.177343724078192, 13.264323219087792, 
     13.220833471582996, 13.133853976573388, 13.046875311061733, 12.872916321042528, 
     12.655468413016461])))
 
    z_top_internal = np.array(list(reversed([14.0 for i in range(len(x_top_internal))])))

    #----------------------------------------------------------
    # Transpose to meters[m]:

    x_top_external = (x_top_external*factor_unit).tolist()
    x_top_internal = (x_top_internal*factor_unit).tolist()
    y_top_external = (y_top_external*factor_unit).tolist()
    y_top_internal = (y_top_internal*factor_unit).tolist()
    z_top_external = (z_top_external*factor_unit).tolist()
    z_top_internal = (z_top_internal*factor_unit).tolist()

    top_internal= [x_top_internal, y_top_internal, z_top_internal]

    top_external = [x_top_external, y_top_external, z_top_external]

    bottom_internal = [deepcopy(x_top_internal), deepcopy(y_top_internal), [0.0 for i in range(len(x_top_internal))]]

    bottom_external = [deepcopy(x_top_external), deepcopy(y_top_external), [0.0 for i in range(len(x_top_external))]]

    # Finally, Calls the principal function to parametrize the disc geometry

    disc_geometry(top_internal, top_external, bottom_internal, bottom_external, 15, mesh_file_name, transfinite_annulus_radial=
     transfinite_annulus_radial, transfinite_circumferential=transfinite_circumferential, transfinite_nucleus_radial=
     transfinite_nucleus_radial, transfinite_axial=transfinite_axial, bias_radial_nucleus=bias_radial_nucleus,
     bias_radial_annulus=bias_radial_annulus, bias_axial=bias_axial, factor_unit=factor_unit)
    
    # |----------------------------------------------------------------------------|
    # |                          LOCAL COORDINATE SYSTEM                           |            
    # |----------------------------------------------------------------------------|
    
    circumferential_direction_file_name = "circumferential_direction_000"

    axial_direction_file_name = "axial_direction_000"

    radial_direction_file_name = "radial_direction_000"

    mesh_data_class = coordinates_direction(circumferential_direction_file_name, 
    axial_direction_file_name, radial_direction_file_name, mesh_file_name, parent_path)

    # |----------------------------------------------------------------------------|
    # |                      GENERATE PARAMETERS COMBINATION SETUP                 |
    # |----------------------------------------------------------------------------|

    n_combination = 600

    properties = [[[0.05, 0.09], "kappa", function_scalar], 
    [[60.54, 61.86], [-0.2, -0.19], [-0.04, -0.03], parent_path, mesh_file_name, mesh_data_class, "k2", function_field],
    [[2.88E6, 3.68E6], [-0.13, -0.1], [-0.15, -0.04], parent_path, mesh_file_name, mesh_data_class, "k1", function_field], 
    [[33.07, 36.73], [0.04, 0.07], [0.03, 0.07], parent_path, mesh_file_name, mesh_data_class, "gamma", function_field],
    [[0.14E6, 0.18E6], "c1", function_scalar], 
    [[0.03E6, 0.03E6], "c2", function_scalar], 
    [[0.19E6, 0.2E6], "c", function_scalar],
    [[-3.1*factor_unit, 3.1*factor_unit], [-3.7*factor_unit, 3.7*factor_unit], 
    [-2.1*factor_unit, 1.5*factor_unit], [-12.0*(np.pi/180), 12.0*(np.pi/180)],
    [-7.0*(np.pi/180), 7.0*(np.pi/180)], [-7.0*(np.pi/180), 7.0*(np.pi/180)], "translation: x,y,z and rotation: x,y,z", function_dirichlet]]


    a, b = generate_paramenter_combination(properties, n_combination, output_file_name=
    "combinations_simulation_"+str(terminal_number), input_file_name_ann="ann_combinations_"+
    str(terminal_number))


    print(a)
    print(b)
    print("\nFINISHED")