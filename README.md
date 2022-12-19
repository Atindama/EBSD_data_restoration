<!-- TODO update this file -->
# EBSD_TVflow
This is work in progress.
It is made up of functions that:

      1. Read EBSD data in ctf file format called read_ctf
      2. Convert Euler angles to Quaternions and vice versa called Eulers_to_Quaternions and Quaternions_to_Eulers respectively. They depend on the functions euler_to_quaternion and quaternion_to_euler
      3. Rescale an array to whatever range you choose. This function is called scale
      4. Estimate the noise variance vector of a given data array
      5. Done: Update Scale to handle nan values
      6. Working on TV flow code
