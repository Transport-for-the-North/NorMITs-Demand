
from normits_demand.distribution import segment_disaggregator as sd


if __name__ == '__main__':

    import_folder = 'Y:/NorMITs Synthesiser/Noham/iter8c/Distribution Outputs/PA Matrices'
    target_tld_folder = 'Y:/NorMITs Synthesiser/import/trip_length_bands/north/enhanced_segments'
    # Using single tld for whole country - run North and GB and compare
    base_hb_productions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Production Outputs/hb_productions_noham.csv'
    base_nhb_productions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Production Outputs/nhb_productions_noham.csv'
    base_hb_attractions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Attraction Outputs/noham_hb_attractions.csv'
    base_nhb_attractions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Attraction Outputs/noham_nhb_attractions.csv'
    export_folder = 'Y:/NorMITs Synthesiser/Noham/iter8c/Distribution Outputs/Segmented Distributions'
    lookup_folder = 'Y:/NorMITs Synthesiser/Noham/Model Zone Lookups'

    base_productions_path = base_hb_productions
    base_attractions_path = base_hb_attractions

    sd_out = sd.disaggregate_segments(
        import_folder,
        target_tld_folder,
        base_productions_path,
        base_attractions_path,
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
