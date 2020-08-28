# ### Constant Values ### #

EFS_COLUMN_DICTIONARY = {
    "base_year_population": [
        "model_zone_id",
        "base_year_population"
    ],
    "base_year_households": [
        "model_zone_id",
        "base_year_households"
    ],
    "base_year_workers": [
        "model_zone_id",
        "base_year_workers"
    ],
    "population": [
        "model_zone_id"
    ],
    "population_ratio": [
        "model_zone_id",
        "property_type_id",
        "traveller_type_id"
    ],
    "households": [
        "model_zone_id"
    ],
    "employment": [
        "model_zone_id"
    ],
    "housing_occupancy": [
        "model_zone_id",
        "property_type_id"
    ],
    "production_trips": [
        "p",
        "traveller_type",
        "soc",
        "ns",
        "area_type"
    ],
    "mode_split": [
        "area_type_id",
        "car_availability_id",
        "purpose_id",
        "mode_id",
    ],
    "mode_time_split": [
        "purpose_id",
        "traveller_type_id",
        "area_type_id",
        "mode_time_split"
    ],
    "employment_ratio": [
        "model_zone_id",
        "employment_class"
    ],
    "attraction_weights": [
        "employment_class",
        "purpose_id"
    ]
}

# ### Default Values ### #

# TODO: What is this area?
DEFAULT_ZONE_SUBSET = [1, 2, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062]

# ### Default Function Arguments ### #

# ## EFS_RUN ## #
# Nested as P, SOC/NS, CA
EFS_RUN_DISTRIBUTIONS_DICT_SYSTRA = {
    1: {
        0: {
            1: "/PA Matrices 24hr/hb_pa_p1_m6_soc0_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p1_m6_soc0_ca2.csv",
            },
        1: {
            1: "/PA Matrices 24hr/hb_pa_p1_m6_soc1_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p1_m6_soc1_ca2.csv",
            },
        2: {
            1: "/PA Matrices 24hr/hb_pa_p1_m6_soc2_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p1_m6_soc2_ca2.csv",
            },
        3: {
            1: "/PA Matrices 24hr/hb_pa_p1_m6_soc3_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p1_m6_soc3_ca2.csv",
            },
    },
    2: {
        0: {
            1: "/PA Matrices 24hr/hb_pa_p2_m6_soc0_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p2_m6_soc0_ca2.csv",
            },
        1: {
            1: "/PA Matrices 24hr/hb_pa_p2_m6_soc1_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p2_m6_soc1_ca2.csv",
            },
        2: {
            1: "/PA Matrices 24hr/hb_pa_p2_m6_soc2_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p2_m6_soc2_ca2.csv",
            },
        3: {
            1: "/PA Matrices 24hr/hb_pa_p2_m6_soc3_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p2_m6_soc3_ca2.csv",
            },
        },
    3: {
        1: {
            1: "/PA Matrices 24hr/hb_pa_p3_m6_ns1_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p3_m6_ns1_ca2.csv",
            },
        2: {
            1: "/PA Matrices 24hr/hb_pa_p3_m6_ns2_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p3_m6_ns2_ca2.csv",
            },
        3: {
            1: "/PA Matrices 24hr/hb_pa_p3_m6_ns3_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p3_m6_ns3_ca2.csv",
            },
        4: {
            1: "/PA Matrices 24hr/hb_pa_p3_m6_ns4_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p3_m6_ns4_ca2.csv",
            },
        5: {
            1: "/PA Matrices 24hr/hb_pa_p3_m6_ns5_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p3_m6_ns5_ca2.csv",
            },
        },
    4: {
        1: {
            1: "/PA Matrices 24hr/hb_pa_p4_m6_ns1_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p4_m6_ns1_ca2.csv",
            },
        2: {
            1: "/PA Matrices 24hr/hb_pa_p4_m6_ns2_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p4_m6_ns2_ca2.csv",
            },
        3: {
            1: "/PA Matrices 24hr/hb_pa_p4_m6_ns3_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p4_m6_ns3_ca2.csv",
            },
        4: {
            1: "/PA Matrices 24hr/hb_pa_p4_m6_ns4_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p4_m6_ns4_ca2.csv",
            },
        5: {
            1: "/PA Matrices 24hr/hb_pa_p4_m6_ns5_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p4_m6_ns5_ca2.csv",
            },
        },
    5: {
        1: {
            1: "/PA Matrices 24hr/hb_pa_p5_m6_ns1_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p5_m6_ns1_ca2.csv",
            },
        2: {
            1: "/PA Matrices 24hr/hb_pa_p5_m6_ns2_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p5_m6_ns2_ca2.csv",
            },
        3: {
            1: "/PA Matrices 24hr/hb_pa_p5_m6_ns3_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p5_m6_ns3_ca2.csv",
            },
        4: {
            1: "/PA Matrices 24hr/hb_pa_p5_m6_ns4_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p5_m6_ns4_ca2.csv",
            },
        5: {
            1: "/PA Matrices 24hr/hb_pa_p5_m6_ns5_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p5_m6_ns5_ca2.csv",
             },
        },
    6: {
        1: {
            1: "/PA Matrices 24hr/hb_pa_p6_m6_ns1_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p6_m6_ns1_ca2.csv",
            },
        2: {
            1: "/PA Matrices 24hr/hb_pa_p6_m6_ns2_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p6_m6_ns2_ca2.csv",
            },
        3: {
            1: "/PA Matrices 24hr/hb_pa_p6_m6_ns3_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p6_m6_ns3_ca2.csv",
            },
        4: {
            1: "/PA Matrices 24hr/hb_pa_p6_m6_ns4_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p6_m6_ns4_ca2.csv",
            },
        5: {
            1: "/PA Matrices 24hr/hb_pa_p6_m6_ns5_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p6_m6_ns5_ca2.csv",
             },
        },
    7: {
        1: {
            1: "/PA Matrices 24hr/hb_pa_p7_m6_ns1_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p7_m6_ns1_ca2.csv",
            },
        2: {
            1: "/PA Matrices 24hr/hb_pa_p7_m6_ns2_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p7_m6_ns2_ca2.csv",
            },
        3: {
            1: "/PA Matrices 24hr/hb_pa_p7_m6_ns3_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p7_m6_ns3_ca2.csv",
            },
        4: {
            1: "/PA Matrices 24hr/hb_pa_p7_m6_ns4_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p7_m6_ns4_ca2.csv",
            },
        5: {
            1: "/PA Matrices 24hr/hb_pa_p7_m6_ns5_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p7_m6_ns5_ca2.csv",
             },
        },
    8: {
        1: {
            1: "/PA Matrices 24hr/hb_pa_p8_m6_ns1_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p8_m6_ns1_ca2.csv",
            },
        2: {
            1: "/PA Matrices 24hr/hb_pa_p8_m6_ns2_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p8_m6_ns2_ca2.csv",
            },
        3: {
            1: "/PA Matrices 24hr/hb_pa_p8_m6_ns3_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p8_m6_ns3_ca2.csv",
            },
        4: {
            1: "/PA Matrices 24hr/hb_pa_p8_m6_ns4_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p8_m6_ns4_ca2.csv",
            },
        5: {
            1: "/PA Matrices 24hr/hb_pa_p8_m6_ns5_ca1.csv",
            2: "/PA Matrices 24hr/hb_pa_p8_m6_ns5_ca2.csv",
        },
    },
}

# nested as P, CA
EFS_RUN_DISTRIBUTIONS_DICT = {
    1: {
        1: "/dist-hb-car/hb_tp_pa_car_availability1_mode3_purpose1.csv",
        2: "/dist-hb-car/hb_tp_pa_car_availability2_mode3_purpose1.csv",
    },
    2: {
        1: "/dist-hb-car/hb_tp_pa_car_availability1_mode3_purpose2.csv",
        2: "/dist-hb-car/hb_tp_pa_car_availability2_mode3_purpose2.csv",
    },
    3: {
        1: "/dist-hb-car/hb_tp_pa_car_availability1_mode3_purpose3.csv",
        2: "/dist-hb-car/hb_tp_pa_car_availability2_mode3_purpose3.csv",
    },
    4: {
        1: "/dist-hb-car/hb_tp_pa_car_availability1_mode3_purpose4.csv",
        2: "/dist-hb-car/hb_tp_pa_car_availability2_mode3_purpose4.csv",
    },
    5: {
        1: "/dist-hb-car/hb_tp_pa_car_availability1_mode3_purpose5.csv",
        2: "/dist-hb-car/hb_tp_pa_car_availability2_mode3_purpose5.csv",
    },
    6: {
        1: "/dist-hb-car/hb_tp_pa_car_availability1_mode3_purpose6.csv",
        2: "/dist-hb-car/hb_tp_pa_car_availability2_mode3_purpose6.csv",
    },
    7: {
        1: "/dist-hb-car/hb_tp_pa_car_availability1_mode3_purpose7.csv",
        2: "/dist-hb-car/hb_tp_pa_car_availability2_mode3_purpose7.csv",
    },
    8: {
        1: "/dist-hb-car/hb_tp_pa_car_availability1_mode3_purpose8.csv",
        2: "/dist-hb-car/hb_tp_pa_car_availability2_mode3_purpose8.csv",
    }
}


