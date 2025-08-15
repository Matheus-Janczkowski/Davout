# Routine to store some tests

import unittest

# Defines a class to test some ready-to-run simulations

class TestANNTools(unittest.TestCase):

    def setUp(self):

        pass

    # Defines a function to test the Cauchy-continuum Neo-Hookean model
    
    def test_cauchyNeoHookean(self):

        print("\n#####################################################"+
        "###################\n#                          Cauchy Neo-Ho"+
        "okean                          #\n###########################"+
        "#############################################\n")

        import aa_tests.hyperelasticity.disc_neo_hookean

    # Defines a function to test the Cauchy-continuum HGO model
    
    def test_cauchyHGO(self):

        print("\n#####################################################"+
        "###################\n#                              Cauchy HG"+
        "O                              #\n###########################"+
        "#############################################\n")

        import aa_tests.hyperelasticity.disc_HGO

    # Defines a function to test the Micropolar-continuum Neo-Hookean 
    # model
    
    def test_micropolarNeoHookean(self):

        print("\n#####################################################"+
        "###################\n#                        Micropolar Neo-"+
        "Hookean                        #\n###########################"+
        "#############################################\n")

        #import tests.micropolar.our_beam_1.beam_with_fibers_macroscale

        pass

    # Defines a function to test the Micropolar-continuum Neo-Hookean 
    # model microscale
    
    def test_micropolarNeoHookeanMicroscale(self):

        print("\n#####################################################"+
        "###################\n#                   Micropolar Neo-Hooke"+
        "an Microscale                  #\n###########################"+
        "#############################################\n")

        #import tests.micropolar.our_beam_1.beam_with_fibers_microscale

        pass

# Runs all tests

if __name__ == "__main__":

    unittest.main()