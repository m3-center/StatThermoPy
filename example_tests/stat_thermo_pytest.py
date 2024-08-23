#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pytest
import stat_thermo as st


'''
Sources:
1. Luchini, G.; Alegre-Requena, J. V.; Funes-Ardoiz, I.; Paton, R. S. GoodVibes: Automated
    Thermochemistry for Heterogeneous Computational Chemistry Data. F1000Research,
    2020, 9, 291 DOI: 10.12688/f1000research.22758.1
    https://github.com/patonlab/GoodVibes/tree/v3.2

2. https://github.com/kkinist/thermo_python/blob/main/chcl3_freq.out
'''

## Differing symmetry numbers
## ethane.out (Gaussian 09, Revision D.01) from reference [1]
## H2O.out (Gaussian 09, Revision D.01) from reference [1]
## CHCl3 (Gaussian 98, Revision A.7,) https://github.com/kkinist/thermo_python/blob/main/chcl3_freq.out [2]
@pytest.mark.parametrize("file, sym_num, temperature, pressure, elements,\
    energy_elec, rotational_const, vibrations,\
    E_w_zpve, U_total, H_total, G_total", [
    ('ethane.out', 1, 298.15, 101325.0, ['C']*2 + ['H']*6,
        -79.8304209466, [80.17177, 19.89003, 19.88986],
        [ 313.8806,  832.5925,  832.9318, 1009.7581, 1235.9432, 1236.1441,
         1433.6862, 1454.4599, 1531.8686, 1532.2036, 1537.4883, 1538.0761,
         3046.9427, 3047.8868, 3098.2497, 3098.3518, 3122.6100, 3122.6885],
        -79.755183, -79.751714, -79.750770, -79.778293),
    ('H2O.out', 2, 298.15, 101325.0, ['H']*2 + ['O'],
        -76.3681281356, [773.58478, 432.16330, 277.26766],
        [1694.8284, 3644.5363, 3778.6962],
        -76.347356, -76.344521, -76.343577, -76.365035),
    ('chcl3_freq.out', 3, 298.15, 101325.0, ['C']*1 + ['H']*1 + ['Cl']*3,
        -1419.27912068, [3.19889, 3.19889, 1.65801],
        [260.9557, 260.9594, 367.3603, 668.1333, 737.8525, 737.8692, 1266.1187, 1266.1213, 3199.0640],
        -1419.259154, -1419.254688, -1419.253743, -1419.287338),
    ])

def test_gaussian(file, sym_num, temperature, pressure, elements,
                  energy_elec, rotational_const, vibrations,
                  E_w_zpve, U_total, H_total, G_total):

    scale_low_vib = 1.0
    scale_high_vib = 1.0

    Ezpve, U, H, G, Cv = st.compute_thermo(elements=elements, sym_num=sym_num, T=temperature, P=pressure,
                                           rotational_const=rotational_const, vibrations=vibrations,
                                           scale_low_vib=scale_low_vib, scale_high_vib=scale_high_vib,
                                           cutoff=1000.0,
                                           energy_elec=energy_elec)

    ## Gaussian 09 results
    precision = 6
    abs_tolerance = 1e-5
    assert E_w_zpve == pytest.approx(round(Ezpve, precision), abs=abs_tolerance)
    assert U_total == pytest.approx(round(U, precision), abs=abs_tolerance)
    assert H_total == pytest.approx(round(H, precision), abs=abs_tolerance)
    assert G_total == pytest.approx(round(G, precision), abs=abs_tolerance)

## nh3 - symmetric top - (Psi4 1.7) # T=298.15 K, P=101325 Pa
## nh4 - spherical top - (Psi4 1.7) # T=298.15 K, P=101325 Pa
## ethanol-psi-f.log (Psi4 1.4.1) # T=298.15 K, P=101325 Pa
## ethanol-psi-f_100.log (Psi4 1.4.1) # T=100.0 K, P=101325 Pa
## ethanol-psi-f_half_atm.log (Psi4 1.4.1) # T=298.15 K, P = 50662.5 Pa
## benzene-psi-f.log (Psi4 1.7) # T=298.15 K, P=101325 Pa
@pytest.mark.parametrize("file, sym_num, temperature, pressure, elements,\
    energy_elec, rotational_const, vibrations,\
    E_w_zpve, U_total, H_total, G_total, Cv_total", [
    ('nh3', 3, 298.15, 101325.0, ['N'] * 1 + ['H'] * 3,
     -56.1843436416442330, [305734.01828 / 1000, 305734.01806 / 1000, 192584.28040 / 1000],
     [1209.0203, 1849.6119, 1849.6119, 3689.4269, 3822.4990, 3822.4990],
     -56.14734012, -56.14448916, -56.14354498, -56.16533317, 0.00988476 / 1000),

    ('nh4', 3, 298.15, 101325.0, ['N'] * 1 + ['H'] * 4,
     -56.1853065186121370, [201303.00807 / 1000, 201303.00807 / 1000, 191786.00493 / 1000],
     [164.9535, 164.9535, 213.3483, 1178.6981, 1846.1958, 1846.1958,
      3694.5164, 3830.0475, 3830.0475],
     -56.14710403, -56.14247538, -56.14153120, -56.16712551, 0.01882857 / 1000),

    ('ethanol-psi-f.log', 1, 298.15, 101325.0, ['C']*2 + ['H']*6 + ['O']*1,
        -154.07567155, [35835.02429/1000, 9481.91523/1000, 8259.89156/1000],
        [269.9690, 316.9352, 447.6171, 886.8606, 977.9822, 1132.9928,
        1217.4698, 1298.8726, 1395.6785, 1424.0477, 1549.7536, 1613.6205,
        1628.8626, 1646.0013, 1686.1682, 3175.7911, 3200.7141, 3212.4326,
        3276.9620, 3289.2754, 4114.6473],
        -153.98964189, -153.98551029, -153.98456611, -154.01489633, 0.02017755/1000),

    ('ethanol-psi-f_100.log', 1, 100.0, 101325.0, ['C']*2 + ['H']*6 + ['O']*1,
        -154.07567155, [35835.02429/1000, 9481.91523/1000, 8259.89156/1000],
        [269.9690, 316.9351, 447.6173, 886.8606, 977.9820, 1132.9927,
        1217.4696, 1298.8726, 1395.6784, 1424.0477, 1549.7537, 1613.6206, 
        1628.8626, 1646.0013, 1686.1681, 3175.7912, 3200.7141, 3212.4334, 
        3276.9620, 3289.2754, 4114.6478],
        -153.98964189, -153.98864748, -153.98833080, -153.99651089, 0.01144056/1000),

    ('ethanol-psi-f_half_atm.log', 1, 298.15, 50662.5, ['C']*2 + ['H']*6 + ['O']*1,
        -154.07567155, [35835.02429/1000, 9481.91523/1000, 8259.89156/1000],
        [269.9690, 316.9352, 447.6172, 886.8606, 977.9819, 1132.9928,
        1217.4700, 1298.8726, 1395.6784, 1424.0477, 1549.7537, 1613.6203,
        1628.8626, 1646.0012, 1686.1682, 3175.7911, 3200.7141, 3212.4328,
        3276.9620, 3289.2754, 4114.6472],
        -153.98964189, -153.98551029, -153.98456611, -154.01555079, 0.02017755/1000),

    ('benzene-psi-f.log', 4, 298.15, 101325.0, ['C']*6 + ['H']*6,
        -231.8098199049188111, [5727.82826/1000, 5727.80943/1000, 2863.90942/1000],
        [402.8664, 403.0297, 608.6433, 608.6783, 672.8572, 688.7116,
        861.8131, 861.8683, 973.2063, 976.6356, 976.7400, 1012.8619,
        1022.0169, 1061.8892, 1062.1605, 1168.7969, 1196.3878, 1196.4001,
        1369.7158, 1457.3063, 1507.0592, 1507.3560, 1637.9456, 1638.3040,
        3199.0624, 3209.9978, 3210.0184, 3225.1113, 3225.4615, 3236.2746],
        -231.70917234, -231.70471973, -231.70377555, -231.73534970, 0.02772908/1000)
    ])

def test_psi4_thermo(file, sym_num, temperature, pressure, elements,
                     energy_elec, rotational_const, vibrations,
                     E_w_zpve, U_total, H_total, G_total, Cv_total):

    scale_low_vib = 1.0
    scale_high_vib = 1.0

    Ezpve, U, H, G, Cv = st.compute_thermo(elements=elements, sym_num=sym_num, T=temperature, P=pressure,
                                           rotational_const=rotational_const, vibrations=vibrations,
                                           scale_low_vib=scale_low_vib, scale_high_vib=scale_high_vib,
                                           cutoff=1000.0,
                                           energy_elec=energy_elec)

    ## Psi4 results
    precision = 8
    abs_tolerance = 1e-7
    assert E_w_zpve == pytest.approx(round(Ezpve, precision), abs=abs_tolerance)
    assert U_total == pytest.approx(round(U, precision), abs=abs_tolerance)
    assert H_total == pytest.approx(round(H, precision), abs=abs_tolerance)
    assert G_total == pytest.approx(round(G, precision), abs=abs_tolerance)
    assert Cv_total == pytest.approx(round(Cv, 11), abs=1e-10)


# def test_psi4_thermo():
#     ''' Testing if the different rotational partition functions give the same answer for each type.
#         Testing if the three equations give the same result - just for personal proof.
#     '''
#     for top_type in ['spherical', 'symmetric', 'asymmetric']:
#         results = st.rotational_contribution(temperature=300.0, rot_constant=[0.101, 0.101, 0.101], symmetry_number=2, top_type=top_type)
#         print(f'\n{top_type}: {results}')
