#!/usr/bin/env python
# -*- coding: utf-8 -*-


ACTIN_REACTIONS = {
    "Dimers": (
        ["Dimerize"], 
        ["Reverse_Dimerize"],
    ),
    "Trimers": (
        ["Trimerize1", "Trimerize2", "Trimerize3"], 
        ["Reverse_Trimerize"],
    ),
    "Trimerize": (
        ["Trimerize1", "Trimerize2", "Trimerize3"], 
        [],
    ),
    "Barbed growth ATP": (
        [
            "Barbed_Growth_Nucleate_ATP1",
            "Barbed_Growth_Nucleate_ATP2",
            "Barbed_Growth_Nucleate_ATP3",
            "Barbed_Growth_ATP11",
            "Barbed_Growth_ATP12",
            "Barbed_Growth_ATP13",
            "Barbed_Growth_ATP21",
            "Barbed_Growth_ATP22",
            "Barbed_Growth_ATP23",
            "Branch_Barbed_Growth_ATP1",
            "Branch_Barbed_Growth_ATP2",
        ],
        [],
    ),
    "Barbed growth ADP": (
        [
            "Barbed_Growth_Nucleate_ADP1",
            "Barbed_Growth_Nucleate_ADP2",
            "Barbed_Growth_Nucleate_ADP3",
            "Barbed_Growth_ADP11",
            "Barbed_Growth_ADP12",
            "Barbed_Growth_ADP13",
            "Barbed_Growth_ADP21",
            "Barbed_Growth_ADP22",
            "Barbed_Growth_ADP23",
            "Branch_Barbed_Growth_ADP1",
            "Branch_Barbed_Growth_ADP2",
        ],
        [],
    ),
    "Pointed growth ATP": (
        [
            "Pointed_Growth_ATP11",
            "Pointed_Growth_ATP12",
            "Pointed_Growth_ATP13",
            "Pointed_Growth_ATP21",
            "Pointed_Growth_ATP22",
            "Pointed_Growth_ATP23",
        ],
        [],
    ),
    "Pointed growth ADP": (
        [
            "Pointed_Growth_ADP11",
            "Pointed_Growth_ADP12",
            "Pointed_Growth_ADP13",
            "Pointed_Growth_ADP21",
            "Pointed_Growth_ADP22",
            "Pointed_Growth_ADP23",
        ],
        [],
    ),
    "Pointed shrink ATP": (
        ["Pointed_Shrink_ATP"], 
        ["Fail_Pointed_Shrink_ATP"],
    ),
    "Pointed shrink ADP": (
        ["Pointed_Shrink_ADP"], 
        ["Fail_Pointed_Shrink_ADP"],
    ),
    "Barbed shrink ATP": (
        ["Barbed_Shrink_ATP"], 
        ["Fail_Barbed_Shrink_ATP"],
    ),
    "Barbed shrink ADP": (
        ["Barbed_Shrink_ADP"], 
        ["Fail_Barbed_Shrink_ADP"],
    ),
    "Hydrolyze actin": (
        ["Hydrolysis_Actin"], 
        ["Fail_Hydrolysis_Actin"],
    ),
    "Hydrolyze arp": (
        ["Hydrolysis_Arp"], 
        ["Fail_Hydrolysis_Arp"],
    ),
    "Nucleotide exchange actin": (
        ["Nucleotide_Exchange_Actin"], 
        [],
    ),
    "Nucleotide exchange arp": (
        ["Nucleotide_Exchange_Arp"], 
        [],
    ),
    "Arp2/3 bind ATP": (
        [
            "Arp_Bind_ATP11",
            "Arp_Bind_ATP12",
            "Arp_Bind_ATP13",
            "Arp_Bind_ATP21",
            "Arp_Bind_ATP22",
            "Arp_Bind_ATP23",
        ],
        ["Cleanup_Fail_Arp_Bind_ATP"],
    ),
    "Arp2/3 bind ADP": (
        [
            "Arp_Bind_ADP11",
            "Arp_Bind_ADP12",
            "Arp_Bind_ADP13",
            "Arp_Bind_ADP21",
            "Arp_Bind_ADP22",
            "Arp_Bind_ADP23",
        ],
        ["Cleanup_Fail_Arp_Bind_ADP"],
    ),
    "Arp2 unbind ATP": (
        ["Arp_Unbind_ATP"], 
        ["Fail_Arp_Unbind_ATP"],
    ),
    "Arp2 unbind ADP": (
        ["Arp_Unbind_ADP"], 
        ["Fail_Arp_Unbind_ADP"],
    ),
    "Branch ATP": (
        ["Barbed_Growth_Branch_ATP"], 
        [],
    ),
    "Branch ADP": (
        ["Barbed_Growth_Branch_ADP"], 
        [],
    ),
    "Debranch ATP": (
        ["Debranch_ATP"], 
        ["Fail_Debranch_ATP"],
    ),
    "Debranch ADP": (
        ["Debranch_ADP"], 
        ["Fail_Debranch_ADP"],
    ),
    "Cap bind": (
        [
            "Cap_Bind11",
            "Cap_Bind12",
            "Cap_Bind13",
            "Cap_Bind21",
            "Cap_Bind22",
            "Cap_Bind23",
        ],
        [],
    ),
    "Cap unbind": (
        ["Cap_Unbind"], 
        ["Fail_Cap_Unbind"],
    ),
}
