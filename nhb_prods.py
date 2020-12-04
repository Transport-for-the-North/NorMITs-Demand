
import efs_constants as consts
import efs_production_generator as pm

from external_forecast_system import ExternalForecastSystem


def main():
    nhb_pm = pm.NhbProductionModel(
        import_home=r'Y:\NorMITs Demand\import',
        export_home=r'Y:\NorMITs Demand\norms_2015\v2_3-EFS_Output\iter1',
        model_name='norms',
    )
    prods = nhb_pm.run()

    print(prods)


def main2():
    # TESTING SCENARIOS INPUTS
    # Running control
    run_base_efs = True
    recreate_productions = False
    recreate_attractions = False

    constrain_population = False

    # Controls I/O
    iter_num = 3
    import_home = "Y:/"
    export_home = "E:/"
    model_name = 'norms_2015'   # Make sure the correct mode is being used!!!

    # UZC Inputs
    pop_growth = r"Y:\NorMITs Demand\inputs\scenarios\EFS inputs\Population\Growth factors\SC04_UZC_pop_growthfactors_EFS.csv"
    emp_growth = r"Y:\NorMITs Demand\inputs\scenarios\EFS inputs\Employment\Growth factors\SC04_UZC_job_growthfactors_EFS.csv"

    pop_con = r""

    # Set up constraints
    if constrain_population:
        constraints = consts.CONSTRAINT_REQUIRED_DEFAULT
    else:
        constraints = [False] * 6

    # ## RUN START ## #
    efs = ExternalForecastSystem(
        iter_num=iter_num,
        model_name=model_name,
        import_home=import_home,
        export_home=export_home,
        pop_growth_path=pop_growth,
        emp_growth_path=emp_growth
    )

    if run_base_efs:
        # Generates HB PA matrices
        efs.run(
            desired_zoning="norms_2015",
            constraint_source="Default",
            recreate_productions=recreate_productions,
            recreate_attractions=recreate_attractions,
            echo_distribution=False,
            constraint_required=constraints
        )


if __name__ == '__main__':
    main2()
