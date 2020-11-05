

from tqdm import tqdm

def calculate_cumulative_growth(inputdata, required_year):
    for year in tqdm(range(2019, required_year + 1), desc='hi', total=required_year-2018):
        previous_year = year - 1
        inputdata[year] = (inputdata[previous_year] * (1 + (inputdata[year] / 100)))

    # find column index of required year
    col_idx = inputdata.columns.get_loc(required_year + 1)
    # df with calculated cols only:
    inputdata = inputdata.iloc[:, :col_idx]

    return inputdata