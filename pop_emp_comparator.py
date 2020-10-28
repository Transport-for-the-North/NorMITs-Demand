# -*- coding: utf-8 -*-
"""
    Module to compare the inputs and outputs of the MSOA population and employment data.
"""

##### IMPORTS #####
# Standard imports
import re
import time
from pathlib import Path

# Third party imports
import pandas as pd

# Local imports
from demand_utilities import utils as du

##### CLASSES #####
class PopEmpComparator:
    """Class to read the input MSOA population (or employment) data and compare it to the output.

    Reads the input values, growth, constraints and type ratios and compares these to the
    output population (or employment) data at various levels of aggregation. Csvs are produced
    with the comparisons for checking.
    """
    ZONE_COL = 'model_zone_id'

    def __init__(self, input_csv: str, growth_csv: str,
                 constraint_csv: str, ratio_csv: str,
                 output_csv: str, data_type: {'population', 'employment'},
                 base_year: str):
        """Reads and checks the csvs required for the comparisons.

        Parameters
        ----------
        input_csv : str
            Path to csv containing the input population (or employment) data for
            the base year at MSOA level.
        growth_csv : str
            Path to the csv containing the input growth values for population (or employment)
            at MSOA level for all output years.
        constraint_csv : str
            Path to the csv containing the input constraints for population (or employment)
            at MSOA level for all output years.
        ratio_csv : str
            Path to the csv containing the ratios between the population (or employment) types
            at MSOA level for all output years.
        output_csv : str
            Path to the output population (or employment) data produced by the
            EFSProductionGenerator (or EFSAttractionGenerator) class at MSOA level.
        data_type : {'population', 'employment'}
            Whether 'population' or 'employment' data is given for the comparison
        base_year : str
            Base year of the model.

        Raises
        ------
        ValueError
            If the data_type parameter isn't 'population' or 'employment' or the base_year
            isn't found as a column in the output csv.
        """
        # FIXME Remove temporary function for testing purposes
        def read_print(csv, **kwargs):
            start = time.perf_counter()
            print(f'Reading {csv} ... ', end='')
            df = du.safe_read_csv(csv, **kwargs)
            print(f'Done in {time.perf_counter() - start:.1f}s')
            return df

        # Check data_type is allowed and initialise variables for that type
        self.data_type = data_type.lower()
        if self.data_type == 'population':
            ratio_cols = [self.ZONE_COL, 'property_type_id', 'traveller_type_id']
            base_yr_col  = 'base_year_population'
        elif self.data_type == 'employment':
            ratio_cols = [self.ZONE_COL, 'employment_class']
            base_yr_col = 'base_year_workers'
        else:
            raise ValueError('data_type parameter should be "population" or "employment" '
                             f'not "{data_type}"')

        # Read the output data and extract years columns
        self.output = read_print(output_csv)
        pat = re.compile(r'\d{4}')
        self.years = [i for i in self.output.columns if pat.match(i.strip())]
        # Check base year is present in output
        self.base_year = base_year
        if str(self.base_year) not in self.years:
            raise ValueError(f'Base year ({self.base_year}) not found in the '
                             f'{self.data_type} output DataFrame.')

        # Read the required columns for input csvs
        self.input_data = read_print(input_csv, skipinitialspace=True)
        self.input_data.rename(columns={base_yr_col: str(self.base_year)}, inplace=True)
        cols = [self.ZONE_COL, *self.years]
        self.growth_data = read_print(growth_csv, skipinitialspace=True, usecols=cols)
        self.constraint_data = read_print(constraint_csv, skipinitialspace=True, usecols=cols)
        self.ratio_data = read_print(ratio_csv, skipinitialspace=True,
                                     usecols=ratio_cols + self.years, nrows=100) # FIXME Remove nrows - using it for testing only

        # Normalise the growth data against the base year
        self.growth_data[self.years] = self.growth_data[self.years].div(
            self.growth_data[str(self.base_year)], axis=0)

        # Initialise output folder path
        self.output_dir = Path(output_csv).parent / f'{data_type} comparisons'.title()
        self.output_dir.mkdir(exist_ok=True)

    def compare_totals(self) -> pd.DataFrame:
        """Compares the input and output column totals and produces a summary of the differences.

        Returns
        -------
        pd.DataFrame
            DataFrame containing differences between the input and output column totals.
        """
        # Calculate totals for each input and output for each year column
        totals = pd.DataFrame({'input total': self.input_data.sum(),
                               'constraint total': self.constraint_data.sum(),
                               'mean growth input': self.growth_data.mean(),
                               'output total': self.output.sum()})
        drop_index = [i for i in totals.index if i not in self.years]
        totals = totals.drop(index=drop_index)
        totals.index.name = 'year'

        # Calculate comparison columns
        totals['constraint difference'] = totals['output total'] - totals['constraint total']
        totals['constraint % difference'] = (totals['output total']
                                             / totals['constraint total']) - 1
        totals['output growth'] = (totals['output total']
                                   / totals.loc[self.base_year, 'output total'])
        totals['growth difference'] = totals['output growth'] - totals['mean growth input']
        return totals

    def compare_msoa_totals(self) -> pd.DataFrame:
        """Compares the input and output values at MSOA level and produces a summary.

        Returns
        -------
        pd.DataFrame
            Differences between the input and output values at MSOA level.
        """
        # Calculate totals for each MSOA on the outputs
        cols = [self.ZONE_COL, *self.years]
        output_msoa = self.output[cols].groupby(self.ZONE_COL, as_index=False).sum()

        # Set index of dataframes to ZONE_COL for concat and rename columns with source
        concat = []
        for nm, df in (('input', self.input_data), ('constraint', self.constraint_data),
                       ('growth', self.growth_data), ('output', output_msoa)):
            df = df.set_index(self.ZONE_COL)
            df.columns = pd.MultiIndex.from_tuples([(i, nm) for i in df.columns])
            concat.append(df)
        msoa_comp = pd.concat(concat, axis=1)

        # Calculate comparison columns
        for yr in self.years:
            msoa_comp[(yr, 'constraint difference')] = (msoa_comp[(yr, 'output')]
                                                        - msoa_comp[(yr, 'constraint')])
            msoa_comp[(yr, 'constraint % difference')] = (msoa_comp[(yr, 'output')]
                                                          / msoa_comp[(yr, 'constraint')]) - 1
            msoa_comp[(yr, 'output growth')] = (msoa_comp[(yr, 'output')]
                                                / msoa_comp[(self.base_year, 'output')])
            msoa_comp[(yr, 'growth difference')] = (msoa_comp[(yr, 'output growth')]
                                                    - msoa_comp[(yr, 'growth')])

        # Sort and flatten index
        msoa_comp = msoa_comp.sort_index(axis=1, level=0, sort_remaining=False)
        msoa_comp.columns = [f'{i} - {j}' for i, j in msoa_comp.columns]
        return msoa_comp

    def compare_sector_totals(self) -> pd.DataFrame:
        # FIXME Placeholder for sector total comparison
        return pd.DataFrame()

    def ratio_comparison(self) -> pd.DataFrame:
        # FIXME Placeholder for ratio comparison
        return pd.DataFrame()

    def write_comparisons(self):
        """Runs each comparison method and writes the output to a csv. """
        # Run all comparisons and save to csvs
        for func, nm in ((self.compare_totals, 'Totals summary'),
                         (self.compare_msoa_totals, 'MSOA totals comparison'),
                         (self.compare_sector_totals, 'Sector totals comparison'),
                         (self.ratio_comparison, 'Ratio comparison')):
            print(f'Producing {self.data_type.capitalize()} {nm.title()}')
            df = func()
            du.safe_dataframe_to_csv(df, self.output_dir / f'{nm}.csv')
        return

##### FUNCTIONS #####
def test():
    """Tests the PopEmpComparator class on data from previous run of EFS. """
    import_loc = Path('Y:/NorMITs Demand/inputs/default')
    # Population csv files
    population_value_file = "population/base_population_2018.csv"
    population_growth_file = "population/future_population_growth.csv"
    population_constraint_file = "population/future_population_values.csv"
    future_population_ratio_file = "traveller_type/traveller_type_splits.csv"
    population_output_file = (r'C:\WSP_Projects\TfN EFS\02 Delivery\EFS Test Run\NorMITs Demand'
                              r'\norms\v2_2-EFS_Output\iter2\Productions\MSOA_population.csv')

    # Employment csv files
    worker_value_file = "employment/base_workers_2018.csv"
    worker_growth_file = "employment/future_workers_growth.csv"
    worker_constraint_file = "employment/future_workers_growth_values.csv"
    worker_ratio_file = "employment/future_worker_splits.csv"
    worker_output_file = (r'C:\WSP_Projects\TfN EFS\02 Delivery\EFS Test Run\NorMITs Demand'
                          r'\norms\v2_2-EFS_Output\iter2\Attractions\MSOA_workers.csv')

    # Compare the population inputs and outputs
    print('Population Comparison', '-' * 50, sep='\n')
    pop_comp = PopEmpComparator(import_loc / population_value_file,
                                import_loc / population_growth_file,
                                import_loc / population_constraint_file,
                                import_loc / future_population_ratio_file,
                                import_loc / population_output_file,
                                'population', '2018')
    pop_comp.write_comparisons()
    # Compare the employment inputs and outputs
    print('Employment Comparison', '-' * 50, sep='\n')
    emp_comp = PopEmpComparator(import_loc / worker_value_file,
                                import_loc / worker_growth_file,
                                import_loc / worker_constraint_file,
                                import_loc / worker_ratio_file,
                                import_loc / worker_output_file,
                                'employment', '2018')
    emp_comp.write_comparisons()

    return


##### MAIN #####
if __name__ == '__main__':
    test()
