# Routine to store functions to retrieve mesh data

import gmsh

########################################################################
#                           Mesh information                           #
########################################################################

# Defines a function to get the mesh's information

def get_meshInfo(entities):

    # Initializes a dictionary of entities in each physical group

    entities_physicalGroups = dict()

    # Iterates through the list of elements

    for e in entities:
        
        # Gets the dimension and the tag of the entity

        dim = e[0]

        tag = e[1]

        # Initializes a list of element types attached to this entity

        element_typesEntity = []

        # Initializes a list of the quantities of each corresponding el-
        # ement type

        element_typesQuantities = []

        # Get the mesh elements for the entity

        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(
        dim, tag)

        for t in range(len(elemTypes)):

            name, dim1, order, numv, parv, _ = gmsh.model.mesh.getElementProperties(
            elemTypes[t])

            element_typesEntity.append(name)

            element_typesQuantities.append(len(elemTags[t]))

        # Verifies if this entity belongs to a physical group

        physicalTags = gmsh.model.getPhysicalGroupsForEntity(dim, 
        tag)

        if len(physicalTags):

            # Iterates through the physical groups to which the entity
            # belongs

            for p in physicalTags:

                # Verifies if this physical group has already been com-
                # piled

                if p in entities_physicalGroups.keys():

                    # Stores the elements type, updating their quantity

                    for type_n in range(len(element_typesEntity)):

                        type = "Quantity of "+element_typesEntity[type_n]

                        if type in entities_physicalGroups[p].keys():

                            entities_physicalGroups[p][type] += (
                            element_typesQuantities[type_n])

                        else:

                            # Initializes the name of the element
                            # type

                            entities_physicalGroups[p][type] = (
                            element_typesQuantities[type_n])

                else:

                    # Gets the name of the physical group

                    name = gmsh.model.getPhysicalName(dim, p)

                    # Initializes another dictionary top count the 
                    # number of elements to each type of element

                    entities_physicalGroups[p] = {"Name":name}

                    # Stores the elements type, updating their quantity

                    for type_n in range(len(element_typesEntity)):

                        type = "Quantity of "+element_typesEntity[type_n]

                        # Initializes the name of the element type

                        entities_physicalGroups[p][type] = (
                        element_typesQuantities[type_n])

    # Initializes a dictionary of element to show the overall quantity 
    # of each element

    overall_elementsQuantities = dict()

    # Shows the physical groups information in a table

    print("###########################################################"+
    "#############\n#                           Mesh information      "+
    "                     #\n#########################################"+
    "###############################\n")

    # Sorts the entities keys

    entities_keys = list(entities_physicalGroups.keys())

    entities_keys.sort()

    for physical_group in entities_keys:

        text = "Physical group "+str(physical_group)

        for key in entities_physicalGroups[physical_group]:

            quantity = entities_physicalGroups[physical_group][key]

            text += " | "+str(key)+": "+str(quantity)

            # Adds the quantity to the overall dictionary of elements

            if key!="Name":

                if key in overall_elementsQuantities.keys():

                    overall_elementsQuantities[key] += quantity

                else:   

                    overall_elementsQuantities[key] = quantity

        # Prints the line

        print(text)

    print("")

    # Prints the overall quantities

    for element in overall_elementsQuantities:

        print(element, "elements in the whole mesh:", 
        overall_elementsQuantities[element])

    print("")