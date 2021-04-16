
import os

from normits_demand.distribution import segment_disaggregator as sd

IMPORT_DRIVE = "I:/"
MODEL_NAME = 'norms'
ITER_NAME = 'iter4'

# Noham == iter8c
# Norms == iter4


def main():
    import_folder = 'I:/NorMITs Demand/import/norms/post_me/tms_seg_pa'
    export_folder = 'I:/NorMITs Demand/import/norms/post_me/tfn_seg_pa'

    synth_home = os.path.join(IMPORT_DRIVE, 'NorMITs Synthesiser')

    # Using single tld for whole country - run North and GB and compare
    lookup_folder = os.path.join(synth_home, MODEL_NAME, 'Model Zone Lookups')

    # Get the productions and attractions
    p_home = os.path.join(synth_home, MODEL_NAME, ITER_NAME, 'Production Outputs')
    a_home = os.path.join(synth_home, MODEL_NAME, ITER_NAME, 'Attraction Outputs')

    if MODEL_NAME.strip().lower() == 'noham':
        base_hb_productions = os.path.join(p_home, 'hb_productions_noham.csv')
        base_nhb_productions = os.path.join(p_home, 'nhb_productions_noham.csv')
        base_hb_attractions = os.path.join(a_home, 'noham_hb_attractions.csv')
        base_nhb_attractions = os.path.join(a_home, 'noham_nhb_attractions.csv')
        target_tld_folder = os.path.join(synth_home, 'import/trip_length_bands/north/enhanced_segments')
    elif MODEL_NAME.strip().lower() == 'norms':
        base_hb_productions = os.path.join(p_home, 'fake out/hb_productions_norms.csv')
        base_nhb_productions = os.path.join(p_home, 'fake out/nhb_productions_norms.csv')
        base_hb_attractions = os.path.join(a_home, 'fake out/ca/norms_hb_attractions.csv')
        base_nhb_attractions = os.path.join(a_home, 'fake out/ca/norms_nhb_attractions.csv')
        target_tld_folder = os.path.join(synth_home, 'import/trip_length_bands/gb/enhanced_ca_segments')
    else:
        raise ValueError(
            "I don't know what the model name '%s' is!" % MODEL_NAME
        )

    # HB conversion
    sd_out = sd.disaggregate_segments(
        import_folder,
        target_tld_folder,
        MODEL_NAME,
        base_hb_productions,
        base_hb_attractions,
        export_folder,
        lookup_folder,
        aggregate_surplus_segments=True,
        rounding=5,
        trip_origin='hb',
        tp='24hr',
        iz_infill=0.5,
        furness_loops=1999,
        min_pa_diff=.1,
        bs_con_crit=.975,
        mp_threads=-1,
        export_original=True,
        export_furness=False)

    # nhb_sd_out = sd.disaggregate_segments(
    #     import_folder,
    #     target_tld_folder,
    #     MODEL_NAME,
    #     base_nhb_productions,
    #     base_nhb_attractions,
    #     export_folder,
    #     lookup_folder,
    #     aggregate_surplus_segments=True,
    #     rounding=5,
    #     trip_origin='nhb',
    #     # tp='tp',
    #     tp='24hr',
    #     iz_infill=0.5,
    #     furness_loops=1999,
    #     min_pa_diff=.1,
    #     bs_con_crit=.975,
    #     mp_threads=-1,
    #     export_original=True,
    #     export_furness=False)


if __name__ == '__main__':
   main()
