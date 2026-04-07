# Based on gsplat; modified from the original.
# Licensed under the Apache License, Version 2.0.


# for compatibility with nerfstudio==1.1.4
def num_sh_bases(degree: int) -> int:
    """
    Returns the number of spherical harmonic bases for a given degree.
    """
    assert degree <= 4, "We don't support degree greater than 4."
    return (degree + 1) ** 2
