
reinitialize
# Where Pymol finds your xyz
load xyz_file, xyz_name

selection

#Praeamble
# Set the diameter of the spheres and the sticks for all atoms.
# note, that the (diameter of the spheres) = spherescale x (van der waals diameter)
set sphere_scale, 0.2
set stick_radius, 0.2
# redefine the color and the spherescale for selected elements
# C
#set sphere_scale, 0.18, elem C
# H
#set sphere_scale, 0.13, elem H
# N
#set sphere_scale, 0.18, elem N


#
bg_color white

# Enable background transparency for the png-Export
show spheres, all
show sticks, all



#view_angle

#Nice angle view:
#set_view (\
#     0.436443806,   -0.784276843,   -0.440924108,\
#    -0.535781920,    0.167149663,   -0.827646375,\
#    -0.722806275,   -0.597458363,    0.347247094,\
#     0.000005078,    0.000004047,  -10.060256958,\
#    -0.170541331,    0.023267522,    0.586957872,\
#     6.429908752,   13.689904213,  -20.000000000 )


#for right angle view: turn y, 50

#angle bending, C,N,H3
#dihedral dihed, H1,C,N,H3

#set angle_label_position, 1.65, bending
#set dihedral_label_position, 1.65, dihed

#set angle_size, 1
#set dihedral_size, 1.2

#set angle_color,red
#set dihedral_color,red

#set label_size, 60
#set label_font, 9

#hide labels, bending
#hide labels, dihed

#util.performance(0)

set ray_opaque_background, 0

png xyz_name.png, width=xyz_width, height=xyz_height, ray=1
