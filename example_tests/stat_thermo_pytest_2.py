# -*- coding: utf-8 -*-

"""Test for extracting raw energies from QM calculations.
    Currently supports GAMESS and Psi4."""

from __future__ import absolute_import, division, print_function

# from .. import helper_functions
# from coffe.core import coffedir
# from coffe.core import pkgdata
# from coffe.quantum import parser_qm

import pytest

from stat_thermo import *

def test_psi4_thermodynamics(tmpdir):
    """Test to reproduce Psi4's thermodynamics.

        This test will likely fail since rounding errors, I think.

        Make sure that the reported differences are small.

        Test calculation: nonane mp2/avdz conf 9 - OB_Conformer-nonane-9-psi-f.log
    """

    EElec = 0.0
    SElec = 0.0
    CElec = 0.0

    ETrans, STrans, CTrans = translational_contribution(temperature=298.15, pressure=101325.00, molecular_mass=128.1565)
    ERot, SRot, CRot = rotational_contribution(temperature=298.15, rot_constant=[0.09433, 0.02127, 0.02054],
                                               symmetry_number=1)

    frequencies = [32.5435, 49.6861, 75.7125, 111.7158, 147.5852, 166.5187, 212.7159,
        240.6705, 254.3465, 292.3040, 314.0136, 363.6690, 418.2049,
        441.0576, 525.1900, 727.5192, 761.3632, 786.8476, 801.4291,
        842.3182, 866.6699, 872.7578, 910.1625, 943.2349, 989.0699,
        1013.8565, 1017.3589, 1060.5926, 1074.7090, 1093.1902, 1120.0292,
        1131.8892, 1159.1036, 1164.8722, 1202.4359, 1242.5322, 1271.8497,
        1283.5483, 1315.0839, 1320.6115, 1335.8123, 1343.3690, 1357.5652,
        1378.2575, 1380.3038, 1385.4169, 1388.2203, 1404.7818, 1415.2974,
        1416.0039, 1486.2126, 1495.2131, 1497.6232, 1502.2143, 1504.9853,
        1506.8623, 1508.3438, 1518.0007, 1520.1188, 1521.9072, 1527.9853,
        3042.5971, 3045.8079, 3046.6485, 3055.8665, 3056.5900, 3061.9295,
        3063.1683, 3067.9612, 3070.0037, 3087.7357, 3094.9060, 3096.9544,
        3108.6856, 3114.4015, 3116.9253, 3120.6226, 3146.3025, 3150.2158,
        3152.1028, 3153.3352]

    VibTemp_all, ZPVE, Evib_sum, Svib_sum, CVib_sum = vibrational_contribution(temperature=298.15, vibrations= \
        frequencies, scaling_factors=[1.0]*len(frequencies))

    Cv = heat_capacity_constant_volume(CvElec=CElec, CvTrans=CTrans, CvRot=CRot, CvVib_sum=CVib_sum)
    S = entropy(STrans=STrans, SElec=SElec, SRot=SRot, Sum_SVib=Svib_sum)
    Ucorr = correction_energy_U(ETrans=ETrans, EElec=EElec, ERot=ERot, Evib_sum=Evib_sum)
    Hcorr = correction_energy_H(temperature=298.15, Ucorr=Ucorr)
    Gcorr = correction_energy_G(temperature=298.15, Hcorr=Hcorr, entropy=S)

    final_thermo = final_thermo_energies(Eelec=-354.20513818, ZPVE=ZPVE, Ucorr=Ucorr, Hcorr=Hcorr,
                                         Gcorr=Gcorr)

    output = [round(ETrans*hartree2kcalmol, 3),   round(STrans*hartree2kcalmol, 6),   round(CTrans*hartree2kcalmol, 6),
              round(ERot*hartree2kcalmol, 3),     round(SRot*hartree2kcalmol, 6),     round(CRot*hartree2kcalmol, 6),
              round(ZPVE*hartree2kcalmol, 3),
              round(Evib_sum*hartree2kcalmol, 3), round(Svib_sum*hartree2kcalmol, 6), round(CVib_sum*hartree2kcalmol, 6),
              round(Cv*hartree2kcalmol, 6),       round(S*hartree2kcalmol, 6),
              round(Ucorr*hartree2kcalmol, 3),    round(Hcorr*hartree2kcalmol, 3),    round(Gcorr*hartree2kcalmol, 3)]

    target_VibTemp = [46.8229, 71.4873, 108.9335,
              160.7342, 212.3423, 239.5833,
              306.0509, 346.2713, 365.9479,
              420.5603, 451.7957, 523.2387,
              601.7037, 634.5836, 755.6314,
              1046.7382, 1095.4321, 1132.0986,
              1153.0781, 1211.9083, 1246.9450,
              1255.7042, 1309.5212, 1357.1050,
              1423.0514, 1458.7138, 1463.7530,
              1525.9566, 1546.2670, 1572.8573,
              1611.4727, 1628.5366, 1667.6921,
              1675.9917, 1730.0376, 1787.7272,
              1829.9086, 1846.7403, 1892.1130,
              1900.0659, 1921.9365, 1932.8089,
              1953.2341, 1983.0057, 1985.9499,
              1993.3065, 1997.3399, 2021.1683,
              2036.2979, 2037.3143, 2138.3290,
              2151.2787, 2154.7464, 2161.3519,
              2165.3387, 2168.0394, 2170.1710,
              2184.0650, 2187.1125, 2189.6856,
              2198.4306, 4377.6199, 4382.2394,
              4383.4488, 4396.7116, 4397.7525,
              4405.4349, 4407.2172, 4414.1131,
              4417.0518, 4442.5641, 4452.8807,
              4455.8278, 4472.7064, 4480.9303,
              4484.5615, 4489.8811, 4526.8287,
              4532.4592, 4535.1742, 4536.9472]

    target_final_energies = [-353.92750650, -353.91561704, -353.91467285, -353.96600869]

    target = [0.889, 0.040457, 0.002981,
              0.889, 0.030049, 0.002981,
              174.217,
              179.900, 0.037539, 0.037224,
              0.043185, 0.108045, 181.677, 182.270, 150.056]

    precision = 4
    abs_tolerance = 2e-3
    for vib_obs, vib_target in zip(VibTemp_all, target_VibTemp):
        assert vib_obs == pytest.approx(round(vib_target, precision), abs=abs_tolerance)

    precision = 9
    abs_tolerance = 1e-7
    for energy_obs, energy_target in zip(final_thermo, target_final_energies):
        assert energy_obs == pytest.approx(round(energy_target, precision), abs=abs_tolerance) 

    assert output == target


def test_psi_thermodynamics_2(tmpdir):
    """Test to reproduce Psi4's thermodynamics.

        Make sure that the reported differences are small.

        Test calculation: ethanol hf/6-31G(d) - ethanol-psi-f.log
    """

    EElec = 0.0
    SElec = 0.0
    CElec = 0.0

    ETrans, STrans, CTrans = translational_contribution(temperature=298.15, pressure=101325.00, molecular_mass=46.0419)
    ERot, SRot, CRot = rotational_contribution(temperature=298.15, rot_constant=[1.19533, 0.31628, 0.27552],
                                               symmetry_number=1)
    frequencies = [269.9690, 316.9350, 447.6175, 886.8606, 977.9841, 1132.9905, 1217.4698,
         1298.8725, 1395.6783, 1424.0477, 1549.7536, 1613.6199, 1628.8626, 1646.0011,
         1686.1690, 3175.7919, 3200.7138, 3212.4304, 3276.9619, 3289.2755, 4114.6453]
    VibTemp_all, ZPVE, Evib_sum, Svib_sum, CVib_sum = vibrational_contribution(temperature=298.15, vibrations= \
        frequencies, scaling_factors=[1.0]*len(frequencies))

    Cv = heat_capacity_constant_volume(CvElec=CElec, CvTrans=CTrans, CvRot=CRot, CvVib_sum=CVib_sum)
    S = entropy(STrans=STrans, SElec=SElec, SRot=SRot, Sum_SVib=Svib_sum)
    Ucorr = correction_energy_U(ETrans=ETrans, EElec=EElec, ERot=ERot, Evib_sum=Evib_sum)
    Hcorr = correction_energy_H(temperature=298.15, Ucorr=Ucorr)
    Gcorr = correction_energy_G(temperature=298.15, Hcorr=Hcorr, entropy=S)
    final_thermo = final_thermo_energies(Eelec=-154.07567155, ZPVE=ZPVE, Ucorr=Ucorr, Hcorr=Hcorr,
                                         Gcorr=Gcorr)

    output = [round(ETrans*hartree2kcalmol, 3),   round(STrans*hartree2kcalmol, 6),   round(CTrans*hartree2kcalmol, 6),
              round(ERot*hartree2kcalmol, 3),     round(SRot*hartree2kcalmol, 6),     round(CRot*hartree2kcalmol, 6),
              round(ZPVE*hartree2kcalmol, 3),
              round(Evib_sum*hartree2kcalmol, 3), round(Svib_sum*hartree2kcalmol, 6), round(CVib_sum*hartree2kcalmol, 6),
              round(Cv*hartree2kcalmol, 6),       round(S*hartree2kcalmol, 6),
              round(Ucorr*hartree2kcalmol, 3),    round(Hcorr*hartree2kcalmol, 3),    round(Gcorr*hartree2kcalmol, 3)]

    target_final_energies = [-153.98964190, -153.98551030, -153.98456612, -154.01489634]

    target = [0.889, 0.037406, 0.002981,
              0.889, 0.022265, 0.002981,
              53.984,
              54.800, 0.004165, 0.006700,
              0.012662, 0.063835, 56.577, 57.170, 38.137]

    # precision = 4
    # abs_tolerance = 2e-3
    # for vib_obs, vib_target in zip(VibTemp_all, target_VibTemp):
    #     assert vib_obs == pytest.approx(round(vib_target, precision), abs=abs_tolerance)

    precision = 9
    abs_tolerance = 4e-8
    for energy_obs, energy_target in zip(final_thermo, target_final_energies):
        assert energy_obs == pytest.approx(round(energy_target, precision), abs=abs_tolerance) 

    assert output == target


# def test_gammes_thermodynamics(tmpdir):
#     """Test to reproduce GAMESS's thermodynamics.
#         Make sure that the reported differences are small.

#         Test calculation: ethanol hf/6-31G(d) - ethanol-gam-f.log

#         Rot. Const.
#         v(cm-1) = ((v(GHz)*1E9)/speed-of-light)/100
#         (v(GHz)*1000/psi_hartree2MHz*psi_hartree2wavenumbers)
#         35.80192 GHz = 1.1942235 cm-1  
#         9.47365      = 0.3160069
#         8.25264      = 0.2752784
#     """

#     EElec = 0.0
#     SElec = 0.0
#     CElec = 0.0

#     ETrans, STrans, CTrans = translational_contribution(temperature=298.15, pressure=101325.00, molecular_mass=46.0418300)
#     ERot, SRot, CRot = rotational_contribution(temperature=298.15, rot_constant=[1.1942235, 0.3160069, 0.2752784],
#                                                symmetry_number=1)
    
#     frequencies = [270.04, 316.77, 447.57, 886.76, 978.0, 1132.96, 1217.55, 1298.74, 1395.56,
#                    1423.95, 1549.5, 1613.43, 1628.73, 1645.83, 1685.95, 3175.36, 3200.27,
#                    3212.21, 3276.73, 3289.02, 4114.61]

#     VibTemp_all, ZPVE, Evib_sum, Svib_sum, CVib_sum = vibrational_contribution(temperature=298.15, vibrations= \
#         frequencies, scaling_factors=[1.0]*len(frequencies))

#     Cv = heat_capacity_constant_volume(CvElec=CElec, CvTrans=CTrans, CvRot=CRot, CvVib_sum=CVib_sum)
#     S = entropy(STrans=STrans, SElec=SElec, SRot=SRot, Sum_SVib=Svib_sum)
#     Ucorr = correction_energy_U(ETrans=ETrans, EElec=EElec, ERot=ERot, Evib_sum=Evib_sum)
#     Hcorr = correction_energy_H(temperature=298.15, Ucorr=Ucorr)
#     Gcorr = correction_energy_G(temperature=298.15, Hcorr=Hcorr, entropy=S)
#     final_thermo = final_thermo_energies(Eelec=-154.0757448407, ZPVE=ZPVE, Ucorr=Ucorr, Hcorr=Hcorr,
#                                          Gcorr=Gcorr)
#     # Rounding to remove the small discrepancies
#     VibTemp_all = [round(vib, 3) for vib in VibTemp_all]
#     final_thermo_list = [round(value, 8) for value in final_thermo]

#     output = [round(ETrans*hartree2kcalmol, 3),   round(STrans*hartree2kcalmol, 6),   round(CTrans*hartree2kcalmol, 6),
#               round(ERot*hartree2kcalmol, 3),     round(SRot*hartree2kcalmol, 6),     round(CRot*hartree2kcalmol, 6),
#               round(ZPVE*hartree2kcalmol, 3),
#               round(Evib_sum*hartree2kcalmol, 3), round(Svib_sum*hartree2kcalmol, 6), round(CVib_sum*hartree2kcalmol, 6),
#               round(Cv*hartree2kcalmol, 6),       round(S*hartree2kcalmol, 6),
#               round(Ucorr*hartree2kcalmol, 3),    round(Hcorr*hartree2kcalmol, 3),    round(Gcorr*hartree2kcalmol, 3)]
    
#     target_final_energies = [None, None, None, None] ## NEED TO FILL IN

#     target = [0.889, 0.037406, 0.002981,
#               0.889, 0.022264, 0.002981,
#               53.980008,
#               54.795, 0.004165, 0.006701,
#               0.012662, 0.063835, 56.573, 57.165, 38.133]

#     # assert target_final_energies == final_thermo
#     assert output == target
    