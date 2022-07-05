CURRENTLY:

 - end                                                                    + end

        actin#pointed_1     actin#3      actin#2       actin#1    actin#barbed_3     
                    \\      //   \\      // || \\      //   \\      //
                     \\    //     \\    //  ||  \\    //     \\    //
                     actin#2      actin#1   ||   actin#3      actin#2
                                    ||     ||
                                    ||    arp3
                                    ||   //     
                                    ||  //   
                                arp2#branched
                                        \\
                                         \\
                                         actin#branch_1
                                         //
                                        //
                                    actin#2
                                        \\
                                         \\
                                         actin#barbed_3

                                            + end
                                            
GOAL:

 - end                                                                    + end

        actin#pointed_1 ==  actin#3  ==  actin#5   ==  actin#2 == actin#barbed_4     
                   \\      //   \\      // || \\      //   \\      //
                    \\    //     \\    //  ||  \\    //     \\    //
                    actin#2   ==   actin#4 ==||   actin#1   == actin#3
                                    ||     ||
                                    ||    arp3
                                    ||   //     
                                    ||  //   
                                arp2#branched
                                    ||    \\
                                    ||     \\
                                    ||   actin#branch_1
                                    ||   //      ||
                                    ||  //       ||
                                    actin#2      ||
                                        \\       || 
                                         \\      ||
                                         actin#barbed_3

                                            + end
    
TODO:
- test the new actin_number_types 
  with 3 and 5
  make sure it runs without errors and visualizes
- debug using raise Exception() or print() above each line that was errored at, next at 530. 
  - check where the actin_number_types is getting changed to 1 
  - @ line 823 in actin_generator.py, actin_number_types = 5 is used 
  - actin_number_types is probably getting changed in get_monomers bc its output that's being used in get_monomers_for_fiber (actin_number_types = 1)
  - actin_number_types is getting changed to 1 when get_monomers_for_fiber() is getting called in get_monomers() method. it’s happening bc get_monomers_for_fiber() takes 7 args (including actin_number_types) but in def get_monomers(), it’s only given 6 arguments so actin_number_types is being set to 1 (which is what is given for the particles arg, the default for particles is an empty dictionary) 
- run tests: `make build` from root of simularium-models-util
  add more test cases to test_actin_number_types.py
- add bonds: with offset of 2
   1 - 3
   3 - 5
   5 - 2
   2 - 4
   ActinUtil.add_bonds_between_actins()
     add util.add_polymer_bond_1D() for offsets 0 and 2
- test again
  make sure it runs without errors and visualizes
- for adding angles and dihedrals, talk to subcellular modeling group (esp Matt A)
- lucid chart 
