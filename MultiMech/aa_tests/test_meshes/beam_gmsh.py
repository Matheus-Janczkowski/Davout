import gmsh

import numpy as np

def generate_beam1Volume(length_x, length_y, length_z, transfinite_x, 
transfinite_y, transfinite_z, file_name, transfinite=True, lc=0.2):

    gmsh.initialize()

    # Adds the points

    p11 = gmsh.model.geo.addPoint(length_x, 0.0, 0.0, lc)

    p12 = gmsh.model.geo.addPoint(length_x, length_y, 0.0, lc)

    p13 = gmsh.model.geo.addPoint(0.0, length_y, 0.0, lc)

    p14 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)

    p21 = gmsh.model.geo.addPoint(length_x, 0.0, length_z, lc)

    p22 = gmsh.model.geo.addPoint(length_x, length_y, length_z, lc)

    p23 = gmsh.model.geo.addPoint(0.0, length_y, length_z, lc)

    p24 = gmsh.model.geo.addPoint(0.0, 0.0, length_z, lc)

    gmsh.model.geo.synchronize()

    # Adds the lines

    l11 = gmsh.model.geo.addLine(p11, p12)

    l12 = gmsh.model.geo.addLine(p12, p13)

    l13 = gmsh.model.geo.addLine(p13, p14)

    l14 = gmsh.model.geo.addLine(p14, p11)

    l21 = gmsh.model.geo.addLine(p21, p22)

    l22 = gmsh.model.geo.addLine(p22, p23)

    l23 = gmsh.model.geo.addLine(p23, p24)

    l24 = gmsh.model.geo.addLine(p24, p21)

    l121 = gmsh.model.geo.addLine(p11, p21)

    l122 = gmsh.model.geo.addLine(p12, p22)

    l123 = gmsh.model.geo.addLine(p13, p23)

    l124 = gmsh.model.geo.addLine(p14, p24)

    gmsh.model.geo.synchronize()

    # Adds the surfaces

    surface_1 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l11, l12, l13, l14])])

    surface_2 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l11, l122, -l21, -l121])])

    surface_3 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l12, l123, -l22, -l122])])

    surface_4 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l13, l124, -l23, -l123])])

    surface_5 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l14, l121, -l24, -l124])])

    surface_6 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l21, l22, l23, l24])])

    gmsh.model.geo.synchronize()

    # Adds the volume

    volume_1 = gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop([
    surface_1, surface_2, surface_3, surface_4, surface_5, surface_6])])

    gmsh.model.geo.synchronize()

    if transfinite:

        # Makes the lines transfinite

        gmsh.model.geo.mesh.setTransfiniteCurve(l11, transfinite_y)

        gmsh.model.geo.mesh.setTransfiniteCurve(l21, transfinite_y)

        gmsh.model.geo.mesh.setTransfiniteCurve(l13, transfinite_y)

        gmsh.model.geo.mesh.setTransfiniteCurve(l23, transfinite_y)

        gmsh.model.geo.mesh.setTransfiniteCurve(l12, transfinite_x)

        gmsh.model.geo.mesh.setTransfiniteCurve(l22, transfinite_x)

        gmsh.model.geo.mesh.setTransfiniteCurve(l14, transfinite_x)

        gmsh.model.geo.mesh.setTransfiniteCurve(l24, transfinite_x)

        gmsh.model.geo.mesh.setTransfiniteCurve(l121, transfinite_z)

        gmsh.model.geo.mesh.setTransfiniteCurve(l122, transfinite_z)

        gmsh.model.geo.mesh.setTransfiniteCurve(l123, transfinite_z)

        gmsh.model.geo.mesh.setTransfiniteCurve(l124, transfinite_z)

        gmsh.model.geo.synchronize()

        # Makes the surfaces transfinite

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_1)

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_2)

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_3)

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_4, cornerTags=[
        p14, p24, p23, p13])

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_5, cornerTags=[
        p11, p21, p24, p14])

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_6)

        gmsh.model.geo.synchronize()

        # Makes the volume transfinite

        gmsh.model.geo.mesh.setTransfiniteVolume(volume_1)

    gmsh.model.geo.synchronize()
    
    gmsh.model.geo.removeAllDuplicates()

    gmsh.model.geo.synchronize()

    # Adds the physical groups
    
    gmsh.model.addPhysicalGroup(3, [volume_1], 1, name='Generic volume1')
    
    gmsh.model.addPhysicalGroup(2, [surface_1], 2, name='back')
    
    gmsh.model.addPhysicalGroup(2, [surface_2], 3, name='right')
    
    gmsh.model.addPhysicalGroup(2, [surface_3], 4, name='lower')
    
    gmsh.model.addPhysicalGroup(2, [surface_4], 5, name='left')
    
    gmsh.model.addPhysicalGroup(2, [surface_5], 6, name='upper')
    
    gmsh.model.addPhysicalGroup(2, [surface_6], 7, name='front')

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(3)

    gmsh.write(file_name+".msh")
    
    gmsh.fltk.run()

    gmsh.finalize()

def generate_beam2Volumes(length_x, length_y, length_z, transfinite_x, 
transfinite_y, transfinite_z, file_name, transfinite=True, lc=0.2):

    gmsh.initialize()

    # Adds the points

    p11 = gmsh.model.geo.addPoint(length_x, 0.0, 0.0, lc)

    p12 = gmsh.model.geo.addPoint(length_x, length_y, 0.0, lc)

    p13 = gmsh.model.geo.addPoint(0.0, length_y, 0.0, lc)

    p14 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)

    p21 = gmsh.model.geo.addPoint(length_x, 0.0, 0.5*length_z, lc)

    p22 = gmsh.model.geo.addPoint(length_x, length_y, 0.5*length_z, lc)

    p23 = gmsh.model.geo.addPoint(0.0, length_y, 0.5*length_z, lc)

    p24 = gmsh.model.geo.addPoint(0.0, 0.0, 0.5*length_z, lc)

    p31 = gmsh.model.geo.addPoint(length_x, 0.0, length_z, lc)

    p32 = gmsh.model.geo.addPoint(length_x, length_y, length_z, lc)

    p33 = gmsh.model.geo.addPoint(0.0, length_y, length_z, lc)

    p34 = gmsh.model.geo.addPoint(0.0, 0.0, length_z, lc)

    gmsh.model.geo.synchronize()

    # Adds the lines

    l11 = gmsh.model.geo.addLine(p11, p12)

    l12 = gmsh.model.geo.addLine(p12, p13)

    l13 = gmsh.model.geo.addLine(p13, p14)

    l14 = gmsh.model.geo.addLine(p14, p11)

    l21 = gmsh.model.geo.addLine(p21, p22)

    l22 = gmsh.model.geo.addLine(p22, p23)

    l23 = gmsh.model.geo.addLine(p23, p24)

    l24 = gmsh.model.geo.addLine(p24, p21)

    l31 = gmsh.model.geo.addLine(p31, p32)

    l32 = gmsh.model.geo.addLine(p32, p33)

    l33 = gmsh.model.geo.addLine(p33, p34)

    l34 = gmsh.model.geo.addLine(p34, p31)

    l121 = gmsh.model.geo.addLine(p11, p21)

    l122 = gmsh.model.geo.addLine(p12, p22)

    l123 = gmsh.model.geo.addLine(p13, p23)

    l124 = gmsh.model.geo.addLine(p14, p24)

    l231 = gmsh.model.geo.addLine(p21, p31)

    l232 = gmsh.model.geo.addLine(p22, p32)

    l233 = gmsh.model.geo.addLine(p23, p33)

    l234 = gmsh.model.geo.addLine(p24, p34)

    gmsh.model.geo.synchronize()

    # Adds the surfaces

    surface_1 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l11, l12, l13, l14])])

    surface_2 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l11, l122, -l21, -l121])])

    surface_3 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l12, l123, -l22, -l122])])

    surface_4 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l13, l124, -l23, -l123])])

    surface_5 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l14, l121, -l24, -l124])])

    surface_6 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l21, l22, l23, l24])])

    surface_7 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l21, l232, -l31, -l231])])

    surface_8 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l22, l233, -l32, -l232])])

    surface_9 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l23, l234, -l33, -l233])])

    surface_10 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l24, l231, -l34, -l234])])

    surface_11 = gmsh.model.geo.addPlaneSurface([
    gmsh.model.geo.addCurveLoop([l31, l32, l33, l34])])

    gmsh.model.geo.synchronize()

    # Adds the volume

    volume_1 = gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop([
    surface_1, surface_2, surface_3, surface_4, surface_5, surface_6])])

    volume_2 = gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop([
    surface_6, surface_7, surface_8, surface_9, surface_10, surface_11])
    ])

    gmsh.model.geo.synchronize()

    if transfinite:

        # Makes the lines transfinite

        gmsh.model.geo.mesh.setTransfiniteCurve(l11, transfinite_y)

        gmsh.model.geo.mesh.setTransfiniteCurve(l21, transfinite_y)

        gmsh.model.geo.mesh.setTransfiniteCurve(l13, transfinite_y)

        gmsh.model.geo.mesh.setTransfiniteCurve(l23, transfinite_y)

        gmsh.model.geo.mesh.setTransfiniteCurve(l31, transfinite_y)

        gmsh.model.geo.mesh.setTransfiniteCurve(l33, transfinite_y)

        gmsh.model.geo.mesh.setTransfiniteCurve(l12, transfinite_x)

        gmsh.model.geo.mesh.setTransfiniteCurve(l22, transfinite_x)

        gmsh.model.geo.mesh.setTransfiniteCurve(l14, transfinite_x)

        gmsh.model.geo.mesh.setTransfiniteCurve(l24, transfinite_x)

        gmsh.model.geo.mesh.setTransfiniteCurve(l32, transfinite_x)

        gmsh.model.geo.mesh.setTransfiniteCurve(l34, transfinite_x)

        gmsh.model.geo.mesh.setTransfiniteCurve(l121, transfinite_z)

        gmsh.model.geo.mesh.setTransfiniteCurve(l122, transfinite_z)

        gmsh.model.geo.mesh.setTransfiniteCurve(l123, transfinite_z)

        gmsh.model.geo.mesh.setTransfiniteCurve(l124, transfinite_z)

        gmsh.model.geo.mesh.setTransfiniteCurve(l231, transfinite_z)

        gmsh.model.geo.mesh.setTransfiniteCurve(l232, transfinite_z)

        gmsh.model.geo.mesh.setTransfiniteCurve(l233, transfinite_z)

        gmsh.model.geo.mesh.setTransfiniteCurve(l234, transfinite_z)

        gmsh.model.geo.synchronize()

        # Makes the surfaces transfinite

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_1)

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_2)

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_3)

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_4, cornerTags=[
        p14, p24, p23, p13])

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_5, cornerTags=[
        p11, p21, p24, p14])

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_6)

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_7)

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_8)

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_9, cornerTags=[
        p24, p34, p33, p23])

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_10, cornerTags=[
        p21, p31, p34, p24])

        gmsh.model.geo.mesh.setTransfiniteSurface(surface_11)

        gmsh.model.geo.synchronize()

        # Makes the volume transfinite

        gmsh.model.geo.mesh.setTransfiniteVolume(volume_1)

        gmsh.model.geo.mesh.setTransfiniteVolume(volume_2)

    gmsh.model.geo.synchronize()
    
    gmsh.model.geo.removeAllDuplicates()

    gmsh.model.geo.synchronize()

    # Adds the physical groups
    
    gmsh.model.addPhysicalGroup(3, [volume_1], 1, name='Generic volume1')
    
    gmsh.model.addPhysicalGroup(2, [surface_1], 2, name='back')
    
    gmsh.model.addPhysicalGroup(2, [surface_2, surface_7], 3, name='right')
    
    gmsh.model.addPhysicalGroup(2, [surface_3, surface_8], 4, name='lower')
    
    gmsh.model.addPhysicalGroup(2, [surface_4, surface_9], 5, name='left')
    
    gmsh.model.addPhysicalGroup(2, [surface_5, surface_10], 6, name='upper')
    
    gmsh.model.addPhysicalGroup(2, [surface_11], 7, name='front')
    
    gmsh.model.addPhysicalGroup(3, [volume_2], 8, name='Generic volume2')

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(3)

    gmsh.write(file_name+".msh")
    
    gmsh.fltk.run()

    gmsh.finalize()

# Defines a function to generate the beam given the micropolar model's
# parameters

def generate_micropolarBeam(mu, ratio_Lb, beta, gamma, file_name, 
n_volumes, transfinite_x=7, transfinite_y=7, transfinite_z=21, 
transfinite=True, beam_widthX=None, beam_widthY=None, beam_length=None):

    # Evaluate the characteristic length and the corresponding beam's 
    # square cross section dimension

    if beam_widthX==None or beam_widthY==None or beam_length==None:

        L = np.sqrt((beta+gamma)/(2*mu))

        b = L/ratio_Lb

        print("ENtra no if")

        beam_widthX = b*1.0

        beam_widthY = b*1.0

        beam_length = b*10.0

    print(beam_length, beam_widthX, beam_widthY)

    # Generates the mesh

    if n_volumes==1:

        generate_beam1Volume(beam_widthX, beam_widthY, beam_length, 
        transfinite_x, transfinite_y, transfinite_z, file_name, 
        transfinite=transfinite)

    elif n_volumes==2:

        transfinite_z = int((0.5*(transfinite_z-1))+1)

        generate_beam2Volumes(beam_widthX, beam_widthY, beam_length, 
        transfinite_x, transfinite_y, transfinite_z, file_name, 
        transfinite=transfinite)