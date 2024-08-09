# StatThermoPy
A python script for computing statatistical thermodynamic observables. The main function within this script
is called `compute_thermo`, which is detailed below.

    def compute_thermo(elements: list, sym_num: int, T: float, P: float, rotational_const: list,
                   vibrations: list, scale_low_vib: float, scale_high_vib: float, energy_elec: float):
    ''' Compute all thermodynamics - a master function.

        Args
            elements:         elements in molecule
            sym_num:          symmetry number
            T:                temperature (K)
            P:                pressure (Pa)
            rotational_const: rotational constants (GHz)
            vibrations:       vibrational frequencies (cm-1)
            scale_low_vib:    scaling factor for low vibrations
            scale_high_vib:   scaling factor for high vibrations
            energy_elec:      electron energy at the bottom of an energy well (hartree)

        Return
            Ezpve: electronic energy corrected with zero-point vibrational energy (hartree)
            U: thermal energy (hartree)
            H: enthalpy (hartree)
            G: free energy (hartree)
            Cv: Constant volume heat capacity (hartree)
    '''

    mass_dict = {'H': 1.00782503223, 'C': 12.000, 'O': 15.994914619257,
                 'N': 14.0030740044, 'Cl': 34.96885269}

    mass = 0.0
    for atom in elements:
        mass = mass + mass_dict[atom]

    ## Translation
    ETrans, STrans, CvTrans = translational_contribution(temperature=T, pressure=P, molecular_mass=mass)

    ## Rotation
    rot_const = []
    for rot in rotational_const:
        rot_const.append(rot/mhz2wavenumber)

    ERot, SRot, CvRot = rotational_contribution(temperature=T, rot_constant=rot_const, symmetry_number=sym_num)

    ## Vibration scaling
    scale = []
    for v in vibrations:
        if v <= 1000:
            scale.append(scale_low_vib)
        else:
            scale.append(scale_high_vib)

    VibTemp_list, ZPVE, Evib_sum, Svib_sum, CvVib_sum = vibrational_contribution(temperature=T,
                                                                                 vibrations=vibrations,
                                                                                 scaling_factors=scale)

    S_total = entropy(STrans=STrans, SElec=0.0, SRot=SRot, Sum_SVib=Svib_sum)

    Cv = heat_capacity_constant_volume(CvElec=0.0, CvTrans=CvTrans, CvRot=CvRot, CvVib_sum=CvVib_sum)

    ## Corrections
    corr_U = correction_energy_U(ETrans=ETrans, EElec=0.0, ERot=ERot, Evib_sum=Evib_sum)
    corr_H = correction_energy_H(temperature=T, Ucorr=corr_U)
    corr_G = correction_energy_G(temperature=T, Hcorr=corr_H, entropy=S_total)

    ## Final Energies
    Ezpve, U, H, G = final_thermo_energies(Eelec=energy_elec, ZPVE=ZPVE, Ucorr=corr_U, Hcorr=corr_H, Gcorr=corr_G)

    # print(Ezpve, U, H, G, Cv)

    return Ezpve, U, H, G, Cv
