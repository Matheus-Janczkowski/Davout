# Routine to store some classes of methods to post-process solution in-
# side the pseudotime stepping methods

from dolfin import *

from ..post_processes import post_processes_functions as post_functions

########################################################################
#                   Post-process's classes templates                   #
########################################################################

# Defines a class to hold the data provided by the code as a context

class PostProcessContext:

    # Sets the common information provided to the class by the system

    def __init__(self, mesh, constitutive_model, dx, position_vector,
    domain_physGroupsNamesToTags, ds, boundary_physGroupsNamesToTags,
    referential_normal):

        self.mesh = mesh

        self.constitutive_model = constitutive_model
        
        self.dx = dx

        self.position_vector = position_vector
        
        self.domain_physGroupsNamesToTags = domain_physGroupsNamesToTags

        self.ds = ds

        self.boundary_physGroupsNamesToTags = boundary_physGroupsNamesToTags

        self.referential_normal = referential_normal

        # Gets the physical groups from the integration measure inside a
        # try-except box because submeshes do not have physical groups

        try:

            self.physical_groupsList = set(dx.subdomain_data().array())

        except:

            self.physical_groupsList = None

        # Gets the physical groups from the integration measure inside a
        # try-except box because submeshes do not have physical groups

        try:

            self.boundary_physicalGroupsList = set(ds.subdomain_data().array())

        except:

            self.boundary_physicalGroupsList = None

# Defines a template for the post-processes' classes

class PostProcessMethod:

    def __init__(self, initialization_function, update_function,
    additional_information, code_providedInfo):

        # Each class must have four variables: the initialization func-
        # tion, i.e. the function that initializes all concerning varia-
        # bles to the post-process method; the update function, which 
        # runs the process proper; a list of required additional infor-
        # mation names, which will be used to retrieve information given
        # in dictionaries by the user, thus, the names are their keys; 
        # a list of variables given by the code, that are retrieven from
        # the context class

        self.initialization_function = initialization_function

        self.update_function = update_function

        self.additional_information = additional_information

        self.code_providedInfo = code_providedInfo

########################################################################
#                      Post-processing tools list                      #
########################################################################

# Sets a class for the method to save the field

class SaveField(PostProcessMethod):

    def __init__(self, context: PostProcessContext):

        # Initializes the parent template class and already passes to it
        # the initialization function, the update function, the additio-
        # nal data names, and the code-provided information. The additi-
        # onal data names as lists are optional arguments, the first va-
        # lue is the name, and the second one is the default value

        super().__init__(post_functions.initialize_fieldSaving, 
        post_functions.update_fieldSaving, ["directory path", 
        "file name", ["intermediate saving flag", False]], [])

# Sets a class for the method to save the Cauchy stress field

class SaveCauchyStressField(PostProcessMethod):

    def __init__(self, context: PostProcessContext):

        # Initializes the parent template class and already passes to it
        # the initialization function, the update function, the additio-
        # nal data names, and the code-provided information. The additi-
        # onal data names as lists are optional arguments, the first va-
        # lue is the name, and the second one is the default value

        super().__init__(post_functions.initialize_cauchyStressSaving, 
        post_functions.update_cauchyStressSaving, ["directory path", 
        "file name", "polynomial degree"], [context.mesh, 
        context.constitutive_model, context.dx, 
        context.physical_groupsList, context.domain_physGroupsNamesToTags])

# Sets a class for the method to save the couple Cauchy stress field

class SaveCoupleCauchyStressField(PostProcessMethod):

    def __init__(self, context: PostProcessContext):

        # Initializes the parent template class and already passes to it
        # the initialization function, the update function, the additio-
        # nal data names, and the code-provided information. The additi-
        # onal data names as lists are optional arguments, the first va-
        # lue is the name, and the second one is the default value

        super().__init__(post_functions.initialize_coupleCauchyStressSaving, 
        post_functions.update_coupleCauchyStressSaving, ["directory pa"+
        "th", "file name", "polynomial degree"], [context.mesh, 
        context.constitutive_model, context.dx, 
        context.physical_groupsList, context.domain_physGroupsNamesToTags])

# Sets a class for the method to save the first Piola-Kirchhoff stress 
# field

class SaveFirstPiolaStressField(PostProcessMethod):

    def __init__(self, context: PostProcessContext):

        # Initializes the parent template class and already passes to it
        # the initialization function, the update function, the additio-
        # nal data names, and the code-provided information. The additi-
        # onal data names as lists are optional arguments, the first va-
        # lue is the name, and the second one is the default value

        super().__init__(post_functions.initialize_firstPiolaStressSaving, 
        post_functions.update_firstPiolaStressSaving, ["directory path", 
        "file name", "polynomial degree"], [context.mesh, 
        context.constitutive_model, context.dx, 
        context.physical_groupsList, context.domain_physGroupsNamesToTags])

# Sets a class for the method to save the couple first Piola-Kirchhoff
# stress field

class SaveCoupleFirstPiolaStressField(PostProcessMethod):

    def __init__(self, context: PostProcessContext):

        # Initializes the parent template class and already passes to it
        # the initialization function, the update function, the additio-
        # nal data names, and the code-provided information. The additi-
        # onal data names as lists are optional arguments, the first va-
        # lue is the name, and the second one is the default value

        super().__init__(post_functions.initialize_coupleFirstPiolaStressSaving, 
        post_functions.update_coupleFirstPiolaStressSaving, ["director"+
        "y path", "file name", "polynomial degree"], [context.mesh, 
        context.constitutive_model, context.dx, 
        context.physical_groupsList, context.domain_physGroupsNamesToTags])

# Sets a class for the method to save the traction field at the referen-
# tial configuration

class SaveReferentialTractionField(PostProcessMethod):

    def __init__(self, context: PostProcessContext):

        # Initializes the parent template class and already passes to it
        # the initialization function, the update function, the additio-
        # nal data names, and the code-provided information. The additi-
        # onal data names as lists are optional arguments, the first va-
        # lue is the name, and the second one is the default value

        super().__init__(post_functions.initialize_tractionSaving, 
        post_functions.update_referentialTractionSaving, ["directory p"+
        "ath", "file name", "polynomial degree"], [context.mesh, 
        context.constitutive_model, context.ds, 
        context.physical_groupsList, context.domain_physGroupsNamesToTags,
        context.referential_normal])

# Sets a class for the method to save the pressure field in a point

class SavePressureAtPoint(PostProcessMethod):

    def __init__(self, context: PostProcessContext):

        # Initializes the parent template class and already passes to it
        # the initialization function, the update function, the additio-
        # nal data names, and the code-provided information. The additi-
        # onal data names as lists are optional arguments, the first va-
        # lue is the name, and the second one is the default value

        super().__init__(post_functions.initialize_pressureAtPointSaving, 
        post_functions.update_pressureAtPointSaving, ["directory path", 
        "file name", "polynomial degree", "point coordinates", "flag p"+
        "lotting"], [context.mesh, context.constitutive_model, context.dx,  
        context.physical_groupsList, context.domain_physGroupsNamesToTags])

# Sets a class for the method to homogenize a field

class HomogenizeField(PostProcessMethod):

    def __init__(self, context: PostProcessContext):

        # Initializes the parent template class and already passes to it
        # the initialization function, the update function, the additio-
        # nal data names, and the code-provided information. The additi-
        # onal data names as lists are optional arguments, the first va-
        # lue is the name, and the second one is the default value

        super().__init__(post_functions.initialize_fieldHomogenization, 
        post_functions.update_fieldHomogenization, ["directory path", 
        "file name", "subdomain"], [context.dx, 
        context.physical_groupsList, context.domain_physGroupsNamesToTags])

# Sets a class for the method to homogenize the gradient of a field

class HomogenizeFieldsGradient(PostProcessMethod):

    def __init__(self, context: PostProcessContext):

        # Initializes the parent template class and already passes to it
        # the initialization function, the update function, the additio-
        # nal data names, and the code-provided information. The additi-
        # onal data names as lists are optional arguments, the first va-
        # lue is the name, and the second one is the default value

        super().__init__(post_functions.initialize_gradientFieldHomogenization, 
        post_functions.update_gradientFieldHomogenization, ["directory"+
        " path", "file name", "subdomain"], [context.dx, 
        context.physical_groupsList, context.domain_physGroupsNamesToTags])

# Sets a class for the method to homogenize the first Piola-Kirchhoff
# stress tensor

class HomogenizeFirstPiola(PostProcessMethod):

    def __init__(self, context: PostProcessContext):

        # Initializes the parent template class and already passes to it
        # the initialization function, the update function, the additio-
        # nal data names, and the code-provided information. The additi-
        # onal data names as lists are optional arguments, the first va-
        # lue is the name, and the second one is the default value

        super().__init__(post_functions.initialize_firstPiolaHomogenization, 
        post_functions.update_firstPiolaHomogenization, ["directory pa"+
        "th", "file name", "subdomain"], [context.dx, 
        context.physical_groupsList, 
        context.domain_physGroupsNamesToTags, context.constitutive_model])

# Sets a class for the method to homogenize the couple first Piola-
# Kirchhoff stress tensor

class HomogenizeCoupleFirstPiola(PostProcessMethod):

    def __init__(self, context: PostProcessContext):

        # Initializes the parent template class and already passes to it
        # the initialization function, the update function, the additio-
        # nal data names, and the code-provided information. The additi-
        # onal data names as lists are optional arguments, the first va-
        # lue is the name, and the second one is the default value

        super().__init__(post_functions.initialize_coupleFirstPiolaHomogenization, 
        post_functions.update_coupleFirstPiolaHomogenization, ["direct"+
        "ory path", "file name", "subdomain"], [context.dx, 
        context.physical_groupsList, 
        context.domain_physGroupsNamesToTags, context.constitutive_model,
        context.position_vector])

# Sets a class for the method to homogenize the Cauchy stress tensor

class HomogenizeCauchy(PostProcessMethod):

    def __init__(self, context: PostProcessContext):

        # Initializes the parent template class and already passes to it
        # the initialization function, the update function, the additio-
        # nal data names, and the code-provided information. The additi-
        # onal data names as lists are optional arguments, the first va-
        # lue is the name, and the second one is the default value

        super().__init__(post_functions.initialize_cauchyHomogenization, 
        post_functions.update_cauchyHomogenization, ["directory path",
        "file name", "subdomain"], [context.dx, 
        context.physical_groupsList, 
        context.domain_physGroupsNamesToTags, context.constitutive_model])

# Sets a class for the method to homogenize the couple Cauchy stress

class HomogenizeCoupleCauchy(PostProcessMethod):

    def __init__(self, context: PostProcessContext):

        # Initializes the parent template class and already passes to it
        # the initialization function, the update function, the additio-
        # nal data names, and the code-provided information. The additi-
        # onal data names as lists are optional arguments, the first va-
        # lue is the name, and the second one is the default value

        super().__init__(post_functions.initialize_coupleCauchyHomogenization, 
        post_functions.update_coupleCauchyHomogenization, ["directory "+
        "path", "file name", "subdomain"], [context.dx, 
        context.physical_groupsList, 
        context.domain_physGroupsNamesToTags, context.constitutive_model])

# Sets a class for the method to get the first elasticity tensor (dP/dF)

class FirstElasticityTensorAtPoint(PostProcessMethod):

    def __init__(self, context: PostProcessContext):

        # Initializes the parent template class and already passes to it
        # the initialization function, the update function, the additio-
        # nal data names, and the code-provided information. The additi-
        # onal data names as lists are optional arguments, the first va-
        # lue is the name, and the second one is the default value

        super().__init__(post_functions.initialize_firstElasticityTensor, 
        post_functions.update_firstElasticityTensor, ["directory path", 
        "file name", "polynomial degree", "point coordinates", "flag p"+
        "lotting", "voigt notation", "plotting arguments"], [context.mesh, 
        context.constitutive_model, 
        context.dx, context.physical_groupsList, 
        context.domain_physGroupsNamesToTags])

# Sets a class for the method to get the second elasticity tensor (dS/dC)

class SecondElasticityTensorAtPoint(PostProcessMethod):

    def __init__(self, context: PostProcessContext):

        # Initializes the parent template class and already passes to it
        # the initialization function, the update function, the additio-
        # nal data names, and the code-provided information. The additi-
        # onal data names as lists are optional arguments, the first va-
        # lue is the name, and the second one is the default value

        super().__init__(post_functions.initialize_secondElasticityTensor, 
        post_functions.update_secondElasticityTensor, ["directory path", 
        "file name", "polynomial degree", "point coordinates", "flag p"+
        "lotting", "voigt notation", "plotting arguments"], [context.mesh, 
        context.constitutive_model, 
        context.dx, context.physical_groupsList, 
        context.domain_physGroupsNamesToTags])

# Sets a class for the method to get the third elasticity tensor 
# (dsigma/db)

class ThirdElasticityTensorAtPoint(PostProcessMethod):

    def __init__(self, context: PostProcessContext):

        # Initializes the parent template class and already passes to it
        # the initialization function, the update function, the additio-
        # nal data names, and the code-provided information. The additi-
        # onal data names as lists are optional arguments, the first va-
        # lue is the name, and the second one is the default value

        super().__init__(post_functions.initialize_thirdElasticityTensor, 
        post_functions.update_thirdElasticityTensor, ["directory path", 
        "file name", "polynomial degree", "point coordinates", "flag p"+
        "lotting", "voigt notation", "plotting arguments"], [context.mesh, 
        context.constitutive_model, context.dx, 
        context.physical_groupsList, 
        context.domain_physGroupsNamesToTags])