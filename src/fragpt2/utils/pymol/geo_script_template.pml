reinitialize
# Where Pymol finds your xyz
load xyz_file, xyz_name

selection

#Praeamble
# Set the diameter of the spheres and the sticks for all atoms.
# note, that the (diameter of the spheres) = spherescale x (van der waals diameter)
set sphere_scale, 0.2
set stick_radius, 0.2

#
bg_color white

# Enable background transparency for the png-Export
show spheres, all
show sticks, all

set ray_opaque_background, 0