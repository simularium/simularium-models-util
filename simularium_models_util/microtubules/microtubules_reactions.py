#!/usr/bin/env python
# -*- coding: utf-8 -*-


MICROTUBULES_REACTIONS = {
    "Grow": [
        "Grow_GTP",
        "Grow_GDP",
    ],
    "GTP Grow": [
        "Grow_GTP",
    ],
    "GDP Grow": [
        "Grow_GDP",
    ],
    "MT Shrink": [
        "Finish_Shrink",
    ],
    "Failed Shrink": [
        "Fail_Shrink_MT_GTP",
        "Fail_Shrink_MT_GTP",
    ],
    "Lateral Attach": [
        "Start_Attach_GTP1",
        "Start_Attach_GTP2",
        "Start_Attach_GTP3",
        "Start_Attach_GDP",
    ],
    "Lateral Detach": [
        "Detach_GTP",
        "Detach_GDP",
    ],
    "Hydrolyze": [
        "Hydrolyze",
    ],
    "Failed Hydrolyze": [
        "Fail_Hydrolyze",
    ]
}
