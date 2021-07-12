# Segment Definitions

Each segmentation should be defined here to allow ease of access when choosing
a segmentation level to use during development

### lu_pop
Equivalent to the segmentation the Land Use population data is received in.

| Segment Name | Description                                           | Values | Notes |
|--------------|-------------------------------------------------------|--------|-------|
| tfn_tt       | TfN Traveller Type - Extension of NTEM Traveller Type | 1-760  |       |
| tfn_at       | TfN Area Type - Extension of NTEM Area Type           | 1-8    |       |


### pure_demand
The segmentation of NoTEM Production model pure_demand output.

| Segment Name | Description                                           | Values | Notes |
|--------------|-------------------------------------------------------|--------|-------|
| p            | NTEM Purpose                                          | 1-8    |       |
| tfn_tt       | TfN Traveller Type - Extension of NTEM Traveller Type | 1-760  |       |
| tfn_at       | TfN Area Type - Extension of NTEM Area Type           | 1-8    |       |


### pure_demand_reporting
A simplified reporting version of pure_demand segmentation.

| Segment Name | Description                                           | Values | Notes |
|--------------|-------------------------------------------------------|--------|-------|
| p            | NTEM Purpose                                          | 1-8    |       |
| tfn_tt       | TfN Traveller Type - Extension of NTEM Traveller Type | 1-760  |       |
| tfn_agg_at   | An Aggregated version of TfN Area Type   | 1-3    | 1: 1-3, 2: 4-6, 3: 7-8  |