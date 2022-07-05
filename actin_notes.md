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
- debug using raise exception or prints above each line that was errored at, next at 530. 
  - check where the actin_number_types is getting changed to 1 
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
