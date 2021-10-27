#!/usr/bin/env python
# -*- coding: utf-8 -*-


ACTIN_REACTIONS = {
    "Dimerize": [
        "Dimerize",
    ],
    "Reverse Dimerize": [
        "Reverse_Dimerize",
    ],
    "Trimerize": [
        "Finish_Trimerize",
    ],
    "Reverse Trimerize": [
        "Reverse_Trimerize",
    ],
    "Grow Pointed": [
        "Finish_Pointed_Growth",
    ],
    "Shrink Pointed": [
        "Pointed_Shrink_ATP", 
        "Pointed_Shrink_ADP",
    ],
    "Grow Barbed": [
        "Finish_Barbed_growth",
    ],
    "Shrink Barbed": [
        "Barbed_Shrink_ATP", 
        "Barbed_Shrink_ADP",
    ],
    "Hydrolyze Actin": [
        "Hydrolysis_Actin", 
    ],
    "Hydrolyze Arp2/3": [
        "Hydrolysis_Arp", 
    ],
    "Bind ATP (actin)": [
        "Nucleotide_Exchange_Actin", 
    ],
    "Bind ATP (arp2/3)": [
        "Nucleotide_Exchange_Arp", 
    ],
    "Bind Arp2/3": [
        "Finish_Arp_Bind", 
    ],
    "Unbind Arp2/3": [
        "Arp_Unbind_ATP", 
        "Arp_Unbind_ADP",
    ],
    "Start Branch": [
        "Nucleate_Branch",
    ],
    "Debranch": [
        "Debranch_ATP", 
        "Debranch_ADP",
    ],
    "Bind Cap": [
        "Finish_Cap_Bind", 
    ],
    "Unbind Cap": [
        "Cap_Unbind",
    ],
}
