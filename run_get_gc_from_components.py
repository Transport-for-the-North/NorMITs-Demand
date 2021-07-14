
import pandas as pd

from normits_demand.elasticity import generalised_costs as gc

rail_cost = {
    'from_model_zone_id': 'origin',
    'to_model_zone_id': 'destination',
    'AE_cost': 'walk',
    'fare_cost': 'fare',
    'IVT_cost': 'ride',
    'Wait_Actual_cost': 'wait',
    'Interchange_cost': 'num_int',
}

car_cost = {
    'from_model_zone_id': 'origin',
    'to_model_zone_id': 'destination',
    'time': 'time',
    'distance': 'dist',
    'toll': 'toll',
}

VOT_2018_RAIL_BUSINESS = 44.96817
VOT_2018_CAR_BUSINESS = 31.3238
VOC_2018_CAR_BUSINESS = 12.06822

# RUNNING
IN_PATH = r"I:\NorMITs Demand\import\noham\costs\elasticity_model_format\car_costs_p2.csv"
OUT_PATH = r"I:\NorMITs Demand\import\noham\costs\generalised cost\car_gc_p2.csv"

CAR_OR_RAIL = 'car'

if CAR_OR_RAIL == 'rail':
    VOT = VOT_2018_RAIL_BUSINESS
    RENAME = rail_cost
elif CAR_OR_RAIL == 'car':
    VOT = VOT_2018_CAR_BUSINESS
    VOC = VOC_2018_CAR_BUSINESS
    RENAME = car_cost


def main():

    costs = pd.read_csv(IN_PATH)
    costs = costs.rename(columns=RENAME)

    origins = costs['origin'].unique()
    destinations = costs['destination'].unique()

    if CAR_OR_RAIL == 'rail':
        gen_cost = gc.gen_cost_mode(
            costs=costs,
            mode=CAR_OR_RAIL,
            vot=VOT,
        )
    elif CAR_OR_RAIL == 'car':
        gen_cost = gc.gen_cost_mode(
            costs=costs,
            mode=CAR_OR_RAIL,
            vot=VOT,
            voc=VOC,
        )

    print(costs)

    gen_cost = pd.DataFrame(
        data=gen_cost,
        index=origins,
        columns=destinations,
    )

    print(gen_cost)

    gen_cost.to_csv(OUT_PATH)


if __name__ == '__main__':
    main()
