
# # lst = ['a', 'b', 'c']
# # lst2 = []

# # for index, item in enumerate(lst):
# #   lst2.append(item + str(index))


# # print(lst)
# # print(lst2)
# # # print('hello world')

# # def get_all_actin_particle_types():
# #         """
# #         get particle types for actin

# #         Actin filaments are polymers and to encode polarity,there are 5 polymer types. 
# #         These are represented as "actin#N" where N is in [1,5]. At branch points, 
# #         2 particles arp2 and arp3 join the pointed end of a branch to the side 
# #         of its mother filament. Spatially, the types are mapped like so:

# #         - end                                                                    + end

# #         actin#pointed_1     actin#3 - - - actin#2 - - - actin#4 - - actin#barbed_3     
# #                     \\      //   \\      // || \\      //   \\      //
# #                     \\    //     \\    //  ||  \\    //     \\    //
# #                     actin#2      actin#1   ||   actin#5      actin#2
# #                                     ||     ||
# #                                     ||    arp3
# #                                     ||   //     
# #                                     ||  //   
# #                                 arp2#branched
# #                                         \\
# #                                         \\
# #                                         actin#branch_1
# #                                         //
# #                                         //
# #                                     actin#2
# #                                         \\
# #                                         \\
# #                                         actin#barbed_3

# #                                             + end
# #         """
# #         result = [
# #             "actin#free",
# #             "actin#free_ATP",
# #             "actin#new",
# #             "actin#new_ATP",
# #             "actin#branch_1",
# #             "actin#branch_ATP_1",
# #             "actin#branch_barbed_1",
# #             "actin#branch_barbed_ATP_1",
# #         ]
# #         for i in actin_number_range(5):
# #             result += [
# #                 f"actin#{i}",
# #                 f"actin#ATP_{i}",
# #                 f"actin#mid_{i}",
# #                 f"actin#mid_ATP_{i}",
# #                 f"actin#pointed_{i}",
# #                 f"actin#pointed_ATP_{i}",
# #                 f"actin#barbed_{i}",
# #                 f"actin#barbed_ATP_{i}",
# #             ]
# #         return result




    
        


# def get_all_polymer_actin_types(vertex_type):
#         """
#         get a list of all numbered versions of a type
#         (e.g. for "actin#ATP" return
#         ["actin#ATP_1", "actin#ATP_2", "actin#ATP_3"])
#         """
#         spacer = "_"  # this outputs "actin#ATP_1..."
#         if "#" not in vertex_type:
#             spacer = "#"    # this outputs "actin#ATP1..."
#         return [
#             f"{vertex_type}{spacer}1",
#             f"{vertex_type}{spacer}2",
#             f"{vertex_type}{spacer}3",
#             f"{vertex_type}{spacer}4",
#             f"{vertex_type}{spacer}5"
#         ]

# print(get_all_polymer_actin_types("actin#mid_ATP")) #outputs actin#mid_ATP_5, is this what we want or is "actin#mid_ATP5" what we want. bc "actin" outputs actin#5; no "_"


def createStudent(name, age, grades=[]):
    return {
        'name': name,
        'age': age,
        'grades': grades
    }
# stored in chrisley, line 102 returns { 'name':'Chrisley', 'age':15, 'grades':[]}
chrisley = createStudent('Chrisley', 15)
dallas = createStudent('Dallas', 16)

def addGrade(student, grade):
    student['grades'].append(grade)
    # To help visualize the grades we have added a print statement
    print(student['grades'])


addGrade(chrisley, 90)
#append 90 to [] in prev output by accessing 'grades' key in student's dictionary (chrisley in this case)
addGrade(dallas, 100)
