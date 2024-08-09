import math


wavenumber2K = 1.4387770 # 1 cm-1 = 1.4388 K  source: https://physics.nist.gov/cuu/pdf/factors_2010.pdf
hatree2kelvin = 3.1577504E5 #                 source: https://physics.nist.gov/cuu/pdf/factors_2010.pdf
ghz2kelvin =  0.04799243415908 
mhz2wavenumber = 29.9792458 # 1 megahertz = 29.9792458 cm-1
kcal2J = 4184            # 1 kcal/mol = 4184 J

## https://psicode.org/psi4manual/master/autodoc_physconst.html
c  = 2.99792458E+8       # speed of light in m/s
hartree2kcalmol = 627.5094737775374055927342256 # Psi4 1 hartree = 627.5095 kcal/mol; Wikipedia = 627.5094740631; NIST 4.3597447222060E-18 J

h  = 6.62607015E-34  # NIST Planck's constant in Js; Psi4 = 6.62606896E-34
kb = 1.380649E-23    # NIST Boltzmann's constant in J/K; Psi4 = 1.3806504E-23
na = 6.02214076E+23  # NIST Avagodro's Number in mol-1; Psi4 = 6.02214179E23
amu2kg = 1.66053906892E-27 # NIST 1 amu = 1.66E-027 kg; Psi4 = 1.660538782E-27
hartree2J = 4.3597447222060E-18 # NIST 1 hartree = 4.359744E-18 J; Psi4 = 4.359744E-18

############################
## Source: Luchini, G.; Alegre-Requena, J. V.; Funes-Ardoiz, I.; Paton, R. S. GoodVibes: Automated Thermochemistry for
##         Heterogeneous Computational Chemistry Data. F1000Research, 2020, 9, 291 DOI: 10.12688/f1000research.22758.1

# Symmetry numbers for different point groups
pg_sm = {"C1": 1, "Cs": 1, "Ci": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "C7": 7, "C8": 8, "D2": 4, "D3": 6,
         "D4": 8, "D5": 10, "D6": 12, "D7": 14, "D8": 16, "C2v": 2, "C3v": 3, "C4v": 4, "C5v": 5, "C6v": 6, "C7v": 7,
         "C8v": 8, "C2h": 2, "C3h": 3, "C4h": 4, "C5h": 5, "C6h": 6, "C7h": 7, "C8h": 8, "D2h": 4, "D3h": 6, "D4h": 8,
         "D5h": 10, "D6h": 12, "D7h": 14, "D8h": 16, "D2d": 4, "D3d": 6, "D4d": 8, "D5d": 10, "D6d": 12, "D7d": 14,
         "D8d": 16, "S4": 4, "S6": 6, "S8": 8, "T": 6, "Th": 12, "Td": 12, "O": 12, "Oh": 24, "Cinfv": 1, "Dinfh": 2,
         "I": 30, "Ih": 60, "Kh": 1}
############################


def translational_contribution(temperature: float, pressure: float, molecular_mass: float) -> tuple:
    ''' Computes the translation contribution to the molecule's
            1) thermal energy
            2) entropy
            3) constant volume heat capacity

        Source: McQuarrie Eq. 5-5 and 8-1

        Args:
            temperature (Kelvin)
            pressure (Pascals)
            molecular_mass (au)

        Returns
            ETrans (hartree)
            STrans (hartree/K)
            CvTrans (hartree/K)

        Library Requirements
            math
    '''
    if not isinstance(temperature, float):
        raise TypeError(f'Error with temperature: {type(temperature)}')
    elif not isinstance(pressure, float):
        raise TypeError(f'Error with pressure: {type(pressure)}')
    elif not isinstance(molecular_mass, float):
        raise TypeError(f'Error with molecular_mass: {type(molecular_mass)}')
    else:
        mass = (molecular_mass)*amu2kg

        qTrans = math.pow( ((2*math.pi*mass*kb*temperature)/( math.pow(h, 2)) ), 3/2)*(kb*temperature/pressure) # McQuarrie Eq. 5-5, 8-1

        # ETrans = (3/2)*(kb*na)*temperature/kcal2J
        # STrans = (kb*na)*(math.log(qTrans)+1+(3/2))/kcal2J
        # CvTrans = (3/2)*(kb*na)/kcal2J
        # print('KNK kcal2J', ETrans, STrans, CvTrans)

        ETrans = (3/2)*(kb*na)*temperature/hartree2J/na
        STrans = (kb*na)*(math.log(qTrans)+1+(3/2))/hartree2J/na
        CvTrans = (3/2)*(kb*na)/hartree2J/na

        return ETrans, STrans, CvTrans


def rotational_contribution(temperature: float, rot_constant: list, symmetry_number: int, top_type: str='asymmetric') -> tuple:
    ''' Computes the rotational contribution to the molecule's
            1) thermal energy
            2) entropy
            3) constant volume heat capacity

        Source: McQuarrie Eq. 8-16, 8-17, and 8-18

        Args:
            temperature (Kelvin)
            rot_constants A, B and C (cm-1)
            symmetry_number
            top_type (default=asymmetric): spherical, symmetric, or asymmetric

                    McQuarrie Eq. 8-16 for spherical top (A == B == C rotational constants)
                    McQuarrie Eq. 8-17 for symmetric top (A != B == C rotational constants)
                    McQuarrie Eq. 8-18, 19 for asymmetric top (A != B != C rotational constants)

                    The asymmetric top is the generalized one, which the others can be derived from.
                        Thus, one only needs to use this equation.

        Returns
            ERot  (hartree)
            SRot  (hartree/K)
            CvRot (hartree/K)

        Library Requirements
            math
    '''

    if not isinstance(temperature, float):
        raise TypeError(f'Error with temperature: {type(temperature)}')
    elif not isinstance(rot_constant, list):
        raise TypeError(f'Error with rot_constant: {type(rot_constant)}')
    elif not all(isinstance(item, float) for item in rot_constant):
        raise TypeError(f'Error within the rot_constant list - need to be a list of floats')
    elif not isinstance(symmetry_number, int):
        raise TypeError(f'Error with symmetry_number: {type(symmetry_number)}')
    else:
        # # if len(set(rot_constant)) == 1:
        # if top_type == 'spherical':
        #     # spherical
        #     print('Spherical top')
        #     qRot = (math.sqrt(math.pi) / symmetry_number) * math.pow((temperature /(rot_constant[0]*wavenumber2K)), (3/2))
        # elif top_type == 'symmetric':
        #     # symmetric
        #     print('Symmetric top')
        #     qRot = (math.sqrt(math.pi)/symmetry_number) * (temperature/(rot_constant[0]*wavenumber2K)) \
        #                                                 * (math.sqrt(temperature/(rot_constant[2]*wavenumber2K)))
        # elif top_type == 'asymmetric':
        #     # asymmetric
        #     print('Asymmetric top')
        #     qRot = (math.sqrt(math.pi)/symmetry_number) * (math.sqrt(math.pow(temperature, 3)/(rot_constant[0]*wavenumber2K \
        #                                                                                      * rot_constant[1]*wavenumber2K \
        #                                                                                      * rot_constant[2]*wavenumber2K)))
        # else:
        #     raise ValueError(f'Error with symmetry top specification - needs to be "spherical", "symmetric", or "asymmetric".')

        # Bypass all above - use the generalized formula, which is for a asymmetric top
        qRot = (math.sqrt(math.pi)/symmetry_number) * (math.sqrt(math.pow(temperature, 3)/(rot_constant[0]*wavenumber2K \
                                                                                         * rot_constant[1]*wavenumber2K \
                                                                                         * rot_constant[2]*wavenumber2K)))

        SRot = (kb*na)*(math.log(qRot)+(3/2))/hartree2J/na # McQuarrie Eq. 8-22
        ERot = (3/2)*(kb*na)*temperature/hartree2J/na      # McQuarrie Eq. 8-20
        CvRot = (3/2)*(kb*na)/hartree2J/na                 # McQuarrie Eq. 8-21

        return ERot, SRot, CvRot


def vibrational_contribution(temperature: float, vibrations: list, scaling_factors: list):
    ''' Computes the vibrational contribution to the molecule's
            1) Zero point energy
            2) VibTemp_list
            3) Evib_sum: vibrational contribution to the internal energy, which includes the ZPVE correction
                            (via defining the energy to the first vibration)
            4) Svib_sum: vibrational contribution to the internal entropy
            5) CvVib_sum: vibrational contribution to the internal constant volume heat capacity

        Global variables:
            c: speed of light in m/s
            kb: Boltzmann's constant in J/K
            h: Planck's constant in Js
            na: Avagodro's Number in mol-1

        Source:
            1) https://gaussian.com/wp-content/uploads/dl/thermo.pdf

        Args:
            temperature (Kelvin)
            vibrations (cm-1) (floats)
            scaling_factors (floats)

        Returns
            VibTemp_list (K) -> list
            zpve (hartree) -> float
            Evib_sum (hartree) -> float (includes ZPVE)
            Svib_sum (hartree/K) -> float
            CvVib_sum (hartree/K) -> float

        Library Requirements
            math
    '''
    if not isinstance(temperature, float):
        raise TypeError(f'Error with temperature: {type(temperature)}')
    elif not isinstance(vibrations, list):
        raise TypeError(f'Error with vibrations: {type(vibrations)}')
    elif not all(isinstance(item, float) for item in vibrations):
        raise TypeError(f'Error within the vibrations list - need to be a list of floats')
    elif not isinstance(scaling_factors, list):
        raise TypeError(f'Error with scaling_factors: {type(scaling_factors)}')
    elif not all(isinstance(item, float) for item in scaling_factors):
        raise TypeError(f'Error within the scaling_factors list - need to be a list of floats')
    else:
        ## Conversions of wavenumbers (cm-1) to vibrational temperature (K)
        VibTemp_list = []
        ZPVE_list = []

        for freq, scale in zip(vibrations, scaling_factors):
            sfreq = freq*scale
            
            VibTemp = (h*c/kb)*100*sfreq
            VibTemp_list.append(VibTemp)

            # ZPVE_contrib = (0.5*h*c*100*sfreq/hartree2J)*hartree2kcalmol
            ZPVE_contrib = (0.5*h*c*100*sfreq/hartree2J)
            ZPVE_list.append(ZPVE_contrib)

        ## Vibrational contribution to the internal energy (kcal/mol), including ZPVE correction
        Evib_all = []
        for freq in VibTemp_list:
            # Evib = (kb*na)*freq*(0.5+(1/(math.exp(freq/temperature)-1)))/kcal2J
            Evib = (kb*na)*freq*(0.5+(1/(math.exp(freq/temperature)-1)))/hartree2J/na
            Evib_all.append(Evib)

        Evib_sum = math.fsum(Evib_all)

        ## Vibrational contribution to the internal entropy (kcal/mol-K)
        SVib_all = []
        for freq in VibTemp_list:
            # SVib = (kb*na)*((freq/temperature)/(math.exp(freq/temperature)-1)-math.log(1-math.exp(-freq/temperature)))/kcal2J
            SVib = (kb*na)*((freq/temperature)/(math.exp(freq/temperature)-1)-math.log(1-math.exp(-freq/temperature)))/hartree2J/na
            SVib_all.append(SVib)

        Svib_sum = math.fsum(SVib_all)

        ## Vibrational contribution to the internal constant volume heat capacity (kcal/mol-K)
        CvVib_all = []
        for freq in VibTemp_list:
            # CVib = (kb*na)*math.exp(freq/temperature)*math.pow(((freq/temperature)/(math.exp(freq/temperature)-1)), 2)/kcal2J
            CVib = (kb*na)*math.exp(freq/temperature)*math.pow(((freq/temperature)/(math.exp(freq/temperature)-1)), 2)/hartree2J/na
            CvVib_all.append(CVib)

        CvVib_sum = math.fsum(CvVib_all)
        ZPVE = math.fsum(ZPVE_list)

        return VibTemp_list, ZPVE, Evib_sum, Svib_sum, CvVib_sum


def heat_capacity_constant_volume(CvElec: float, CvTrans:float, CvRot:float, CvVib_sum:float) -> float:
    ''' Computes the molecule's constant volumne heat capacity (Cv).

        Source: McQuarrie

        Args:
            CvElec (hartree/K)
            CvTrans (hartree/K)
            CvRot (hartree/K)
            CvVib_sum (hartree/K)

        Returns
            Cv (hartree/K)

        Library Requirements
            math
    '''
    if not isinstance(CvElec, float):
        raise TypeError(f'Error with CvElec: {type(CvElec)}')
    elif not isinstance(CvTrans, float):
        raise TypeError(f'Error with CvTrans: {type(CvTrans)}')
    elif not isinstance(CvRot, float):
        raise TypeError(f'Error with CvRot: {type(CvRot)}')
    elif not isinstance(CvVib_sum, float):
        raise TypeError(f'Error with CvVib_sum: {type(CvVib_sum)}')
    else:
        Cv = math.fsum([CvElec, CvTrans, CvRot, CvVib_sum])

        return Cv


def entropy(STrans: float, SElec:float, SRot:float, Sum_SVib:float) -> float:
    ''' Computes the molecule's entropy.

        Source: McQuarrie

        Args:
            STrans (hartree/K)
            SElec (hartree/K)
            SRot (hartree/K)
            Sum_SVib (hartree/K)

        Returns
            S (hartree/K)

        Library Requirements
            math
    '''
    if not isinstance(STrans, float):
        raise TypeError(f'Error with STrans: {type(STrans)}')
    elif not isinstance(SElec, float):
        raise TypeError(f'Error with SElec: {type(SElec)}')
    elif not isinstance(SRot, float):
        raise TypeError(f'Error with SRot: {type(SRot)}')
    elif not isinstance(Sum_SVib, float):
        raise TypeError(f'Error with Sum_SVib: {type(Sum_SVib)}')
    else:
        S = math.fsum([STrans, SElec, SRot, Sum_SVib])

        return S


def correction_energy_U(ETrans: float, EElec:float, ERot:float, Evib_sum:float) -> float:
    ''' Computes the correction to the thermal energy (U).

        Source: McQuarrie

        Args:
            ETrans (hartree)
            EElec (hartree)
            ERot (hartree)
            Evib_sum (hartree)

        Returns
            Ucorr (hartree)

        Library Requirements
            math
    '''
    if not isinstance(ETrans, float):
        raise TypeError(f'Error with ETrans: {type(ETrans)}')
    elif not isinstance(EElec, float):
        raise TypeError(f'Error with EElec: {type(EElec)}')
    elif not isinstance(ERot, float):
        raise TypeError(f'Error with ERot: {type(ERot)}')
    elif not isinstance(Evib_sum, float):
        raise TypeError(f'Error with Evib_sum: {type(Evib_sum)}')
    else:
        Ucorr = math.fsum([ETrans, EElec, ERot, Evib_sum])

        return Ucorr


def correction_energy_H(temperature: float, Ucorr: float) -> float:
    ''' Computes the correction to the enthalpy (H).

        Source: McQuarrie

        Args:
            temperature (K)
            Ucorr (hartree)

        Returns
            Hcorr (hartree)
    '''
    if not isinstance(temperature, float):
        raise TypeError(f'Error with temperature: type({temperature})')
    elif not isinstance(Ucorr, float):
        raise TypeError(f'Error with Ucorr: {type(Ucorr)}')
    else:
        # Hcorr = Ucorr + (na*kb)*temperature/kcal2J  # kcal/mol
        Hcorr = Ucorr + (na*kb)*temperature/hartree2J/na
        return Hcorr


def correction_energy_G(temperature: float, Hcorr: float, entropy: float) -> float:
    ''' Computes the correction to the Gibbs free energy (G).

        Source: McQuarrie

        Args:
            temperature (K)
            Hcorr (hartree)
            entropy (hartree/K)

        Returns
            Gcorr (hartree)
    '''
    if not isinstance(temperature, float):
        raise TypeError(f'Error with temperature: {type(temperature)}')
    elif not isinstance(Hcorr, float):
        raise TypeError(f'Error with Hcorr: {type(Hcorr)}') 
    elif not isinstance(entropy, float):
        raise TypeError(f'Error with entropy: {type(entropy)}')
    else:
        Gcorr = Hcorr - temperature*(entropy)

        return Gcorr


def final_thermo_energies(Eelec: float, ZPVE: float, Ucorr: float, Hcorr: float, Gcorr: float) -> list:
    ''' Computes the following energies:

        1) ZPVE corrected electronic energy (Eo)
        2) Themal corrected internal energy (U)
        3) Enthalpy (H)
        4) Gibbs free energy (G)

        Source: McQuarrie

        Args:
            Eelec (hartree)
            ZPVE  (hartree)
            Ucorr (hartree)
            Hcorr (hartree)
            Gcorr (hartree)

        Returns
            List (hartree): [Eo, U, H, G]  List of corrected final energies
    '''
    if not isinstance(Eelec, float):
        raise TypeError(f'Error with Eelec: {type(Eelec)}')
    elif not isinstance(ZPVE, float):
        raise TypeError(f'Error with ZPVE: {type(ZPVE)}') 
    elif not isinstance(Ucorr, float):
        raise TypeError(f'Error with Ucorr: {type(Ucorr)}')
    elif not isinstance(Hcorr, float):
        raise TypeError(f'Error with Hcorr: {type(Hcorr)}')
    elif not isinstance(Gcorr, float):
        raise TypeError(f'Error with Gcorr: {type(Gcorr)}')
    else:
        Eo = Eelec + (ZPVE)
        U  = Eelec + (Ucorr)
        H  = Eelec + (Hcorr)
        G  = Eelec + (Gcorr)

        return Eo, U, H, G


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