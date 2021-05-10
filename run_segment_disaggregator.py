
import os
import shutil

from normits_demand.utils import file_ops
from normits_demand.distribution import segment_disaggregator as sd

IMPORT_DRIVE = "I:/"
MODEL_NAME = 'norms_2015'
ITER_NAME = 'iter4'

# Noham == iter8c
# Norms == iter4

# BACKLOG: Adjust output of TST so EFS can take in as is
#  labels: EFS, QoL Updates


def main():
    global MODEL_NAME
    import_folder = r'I:\NorMITs Synthesiser\Norms_2015\iter4\Norms 15 Post ME\tms_seg_pa'
    export_folder = r'I:\NorMITs Synthesiser\Norms_2015\iter4\Norms 15 Post ME\tfn_seg_pa'

    synth_home = os.path.join(IMPORT_DRIVE, 'NorMITs Synthesiser')

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
    elif MODEL_NAME.strip().lower() == 'norms_2015':
        base_hb_productions = os.path.join(p_home, 'hb_productions_norms_2015.csv')
        base_nhb_productions = os.path.join(p_home, 'nhb_productions_norms_2015.csv')
        base_hb_attractions = os.path.join(a_home, 'hb_attractions_norms_2015.csv')
        base_nhb_attractions = os.path.join(a_home, 'nhb_attractions_norms_2015.csv')
        target_tld_folder = os.path.join(synth_home, 'import/trip_length_bands/gb/enhanced_ca_segments')
        MODEL_NAME = 'norms'
    else:
        raise ValueError(
            "I don't know what the model name '%s' is!" % MODEL_NAME
        )

    # Using single tld for whole country - run North and GB and compare
    lookup_folder = os.path.join(synth_home, MODEL_NAME, 'Model Zone Lookups')

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
        max_bs_loops=200,
        mp_threads=-1,
        export_original=True,
        export_furness=False)

    nhb_sd_out = sd.disaggregate_segments(
        import_folder,
        target_tld_folder,
        MODEL_NAME,
        base_nhb_productions,
        base_nhb_attractions,
        export_folder,
        lookup_folder,
        aggregate_surplus_segments=True,
        rounding=5,
        trip_origin='nhb',
        # tp='tp',
        tp='24hr',
        iz_infill=0.5,
        furness_loops=1999,
        min_pa_diff=.1,
        bs_con_crit=.975,
        max_bs_loops=200,
        mp_threads=-1,
        export_original=True,
        export_furness=False
    )


def rename_tst_outputs():
    mat_dir = r'I:\NorMITs Synthesiser\Norms_2015\iter4\Norms 15 Post ME\tfn_seg_pa'
    ca_vals = [1, 2]
    soc_vals = [0, 1, 2, 3]
    ns_vals = [1, 2, 3, 4, 5]

    # ## MOVE REPORTS INTO SUB FOLDER ## #
    report_dir = os.path.join(mat_dir, 'reports')
    file_ops.create_folder(report_dir, verbose=False)

    # Find all the reports and move
    csvs = [x for x in os.listdir(mat_dir) if '.csv' in x]
    reports = [x for x in csvs if 'report' in x]

    for report in reports:
        src = os.path.join(mat_dir, report)
        dst = os.path.join(report_dir, report)
        shutil.move(src, dst)

    # ## RENAME THE MATS ## #
    mats = [x for x in os.listdir(mat_dir) if '.csv' in x]

    for old_name in mats:
        new_name = None

        # Add in the year
        mid_name = old_name.replace('_enhpa_', '_pa_yr2018_')

        # Swap soc/ns and ca
        for ca in ca_vals:
            ca_str = 'ca%s' % ca
            if ca_str not in mid_name:
                continue

            for s in soc_vals:
                soc_str = 'soc%s' % s
                if soc_str in mid_name:
                    old = '_%s_%s' % (ca_str, soc_str)
                    new = '_%s_%s' % (soc_str, ca_str)
                    new_name = mid_name.replace(old, new)

            for n in ns_vals:
                ns_str = 'ns%s' % n
                if ns_str in mid_name:
                    old = '_%s_%s' % (ca_str, ns_str)
                    new = '_%s_%s' % (ns_str, ca_str)
                    new_name = mid_name.replace(old, new)

        if new_name is None:
            raise ValueError("Couldn't rename %s" % old_name)

        src = os.path.join(mat_dir, old_name)
        dst = os.path.join(mat_dir, new_name)
        shutil.move(src, dst)


if __name__ == '__main__':
    main()
    # rename_tst_outputs()
