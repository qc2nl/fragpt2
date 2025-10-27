#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:21:01 2024

@author: emielkoridon
"""

import os
import subprocess
import numpy as np


# XYZ_WIDTH = '1920'
# XYZ_HEIGHT = '1080'


def xyzfile_to_geometry(filename):
    with open(filename, "r") as f:
        f.readline()
        f.readline()
        geometry = [(line.split()[0],
                     (float(line.split()[1]), float(line.split()[2]), float(line.split()[3])))
                    for line in f.readlines()]
        f.close()
    return geometry


def xyzfile_to_png(filename, xyz_name=None, output_dir=None,
                   xyz_width=1920, xyz_height=1080):
    """
    Create an .png image of molecular geometry given .xyz file.
    The user might want to adjust the view in PyMol.
    """
    filename = os.path.abspath(filename)
    # print(filename)

    original_dir = os.getcwd()
    if output_dir is None:
        output_dir = original_dir
    else:
        output_dir = os.path.abspath(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if xyz_name is None:
        xyz_name = filename.replace('.xyz', '')

    script_template = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   'geo_to_png_script_template.pml')

    script_output = f'{xyz_name}.pml'

    geometry = xyzfile_to_geometry(filename)

    os.chdir(output_dir)
    fin = open(script_template, 'r')
    fout = open(script_output, 'w')

    for line in fin:
        if 'selection' in line:
            selection = []
            for i, [atom_label, xyz] in enumerate(geometry):
                selection.append(f'sele {atom_label}{i+1}, {xyz_name}///UNK`{i+1}/{atom_label}')
            fout.write(
                line.replace(
                    'selection', '\n'.join(selection)))
        else:
            fout.write(line.replace('xyz_file', filename).replace(
                'xyz_name', xyz_name).replace(
                    'xyz_width', str(xyz_width)).replace(
                        'xyz_height', str(xyz_height)))
    fin.close()
    fout.close()

    subprocess.run(args=['pymol', '-c', script_output])

    os.chdir(original_dir)


def xyzfile_to_script(filename, xyz_name=None, output_dir=None,
                      xyz_width=1920, xyz_height=1080):
    """
    Create an .png image of molecular geometry given .xyz file.
    The user might want to adjust the view in PyMol.
    """
    filename = os.path.abspath(filename)
    # print(filename)

    original_dir = os.getcwd()
    if output_dir is None:
        output_dir = original_dir
    else:
        output_dir = os.path.abspath(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if xyz_name is None:
        xyz_name = filename.replace('.xyz', '')

    script_template = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   'geo_script_template.pml')

    script_output = f'{xyz_name}.pml'

    geometry = xyzfile_to_geometry(filename)

    os.chdir(output_dir)
    fin = open(script_template, 'r')
    fout = open(script_output, 'w')

    for line in fin:
        if 'selection' in line:
            selection = []
            for i, [atom_label, xyz] in enumerate(geometry):
                selection.append(f'sele {atom_label}{i+1}, {xyz_name}///UNK`{i+1}/{atom_label}')
            fout.write(
                line.replace(
                    'selection', '\n'.join(selection)))
        else:
            fout.write(line.replace('xyz_file', filename).replace(
                'xyz_name', xyz_name).replace(
                    'xyz_width', str(xyz_width)).replace(
                        'xyz_height', str(xyz_height)))
    fin.close()
    fout.close()

    output_path = os.path.abspath(script_output)

    os.chdir(original_dir)

    return output_path


def xyzname_to_png_script(xyz_name, output_dir=None,
                          xyz_width=1920, xyz_height=1080):
    """
    Create an .png image of molecular geometry given .xyz file.
    The user might want to adjust the view in PyMol.
    """

    original_dir = os.getcwd()
    if output_dir is None:
        output_dir = original_dir
    else:
        output_dir = os.path.abspath(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    script_template = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   'png_template.pml')

    script_output = f'{xyz_name}_makepng.pml'

    os.chdir(output_dir)

    fin = open(script_template, 'r')
    fout = open(script_output, 'w')

    for line in fin:
        fout.write(line.replace(
            'output_folder', output_dir).replace(
            'xyz_name', xyz_name).replace(
            'xyz_width', str(xyz_width)).replace(
            'xyz_height', str(xyz_height)))
    fin.close()
    fout.close()

    output_path = os.path.abspath(script_output)
    os.chdir(original_dir)

    return output_path


if __name__ == '__main__':
    print(xyzfile_to_geometry('../../../../examples/water_ammonia/water-ammonia.xyz'))
    xyzfile_to_png('../../../../examples/water_ammonia/water-ammonia.xyz',
                   xyz_name='test', output_dir=os.path.join(
                       os.getcwd(), 'tmp'))
