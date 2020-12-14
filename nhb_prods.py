
import efs_constants as consts
import efs_production_generator as pm

from external_forecast_system import ExternalForecastSystem


def main():
    # TESTING SCENARIOS INPUTS
    # Running control
    run_base_efs = True
    recreate_productions = False
    recreate_attractions = True

    run_nhb = True

    constrain_population = False

    # Controls I/O
    iter_num = 2
    import_home = "Y:/"
    export_home = "E:/"
    model_name = 'norms_2015'   # Make sure the correct mode is being used!!!

    # UZC Inputs
    pop_growth = r"Y:\NorMITs Demand\import\scenarios\SC04_UZC\population\future_growth_factors.csv"
    emp_growth = r"Y:\NorMITs Demand\import\scenarios\SC04_UZC\employment\future_growth_factors.csv"

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
    main()
