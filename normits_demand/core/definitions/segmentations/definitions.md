# Segment Definitions

Each segmentation should be defined here to allow ease of access when choosing
a segmentation level to use during development

## NoTEM

#### hb_notem_output
The output segmentation of the home based NoTEM models.
Very similar to `nhb_notem_output`, but this segmentation only covers the 
non-home based purposes.
Output of NoTEM treats all 6 time periods as total values, and NOT averages.

| Segment Name   | Description                                           | Values   | Notes |
|----------------|-------------------------------------------------------|----------|-------|
| `p`            | NTEM Purpose                                          | 1-8      |       |
| `m`            | NTEM Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6 |       |
| `g`            | NTEM Gender.                                          | 1-3      | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                     | 1-3      | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                          | 1-5      |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more              | 1-2      |       |
| `tp`           | NTEM Time Period.                                     | 1-6      |       |


#### hb_notem_output_no_tp
Very similar to `hb_notem_output`, but with no tp segments

| Segment Name   | Description                                           | Values   | Notes |
|----------------|-------------------------------------------------------|----------|-------|
| `p`            | NTEM Purpose                                          | 1-8      |       |
| `m`            | NTEM Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6 |       |
| `g`            | NTEM Gender.                                          | 1-3      | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                     | 1-3      | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                          | 1-5      |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more              | 1-2      |       |


#### nhb_notem_output
The output segmentation of the home based NoTEM models.
Very similar to `hb_notem_output`, but this segmentation only covers the home 
based purposes.
Output of NoTEM treats all 6 time periods as total values, and NOT averages.
 

| Segment Name   | Description                                               | Values     | Notes |
|----------------|-----------------------------------------------------------|------------|-------|
| `p`            | NTEM NHB Purpose                                          | 12-16, 18  |       |
| `m`            | NTEM NHB Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6   |       |
| `g`            | NTEM Gender.                                              | 1-3        | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                         | 1-3        | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                              | 1-5        |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more                  | 1-2        |       |
| `tp`           | NTEM Time Period.                                         | 1-6        |       |


### NoTEM HB Productions

#### notem_lu_pop
Equivalent to the segmentation the Land Use population data as received in.

| Segment Name | Description                                           | Values | Notes |
|--------------|-------------------------------------------------------|--------|-------|
| `tfn_tt`     | TfN Traveller Type - Extension of NTEM Traveller Type | 1-760  |       |
| `tfn_at`     | TfN Area Type - Extension of NTEM Area Type           | 1-8    |       |


#### notem_hb_productions_pure
The segmentation of pure demand output of NoTEM HB Production Model

| Segment Name | Description                                           | Values | Notes |
|--------------|-------------------------------------------------------|--------|-------|
| `p`          | NTEM Purpose                                          | 1-8    |       |
| `tfn_tt`     | TfN Traveller Type - Extension of NTEM Traveller Type | 1-760  |       |
| `tfn_at`     | TfN Area Type - Extension of NTEM Area Type           | 1-8    |       |


#### notem_hb_productions_pure_report
A simplified reporting version of notem_pure_hb_productions segmentation.

| Segment Name | Description                                           | Values | Notes                   |
|--------------|-------------------------------------------------------|--------|-------------------------|
| `p`          | NTEM Purpose                                          | 1-8    |                         |
| `tfn_tt`     | TfN Traveller Type - Extension of NTEM Traveller Type | 1-760  |                         |
| `tfn_agg_at` | An Aggregated version of TfN Area Type                | 1-3    | 1: 1-3, 2: 4-6, 3: 7-8  |


#### notem_hb_productions_full_tfnat
The segmentation of the production mode-time splits. This is the most detailed segmentation
available - and is usually too much for most operations when combined with a 
full zoning system.
As zones are usually directly linked to an area type, this 
segmentation contains a lot of redundant 0s;
often `notem_hb_productions_full` is better used.

| Segment Name   | Description                                           | Values   | Notes |
|----------------|-------------------------------------------------------|----------|-------|
| `p`            | NTEM Purpose                                          | 1-8      |       |
| `tfn_tt`       | TfN Traveller Type - Extension of NTEM Traveller Type | 1-760    |       |
| `tfn_at`       | TfN Area Type - Extension of NTEM Area Type           | 1-8      |       |
| `m`            | NTEM Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6 |       |
| `tp`           | NTEM Time Period.                                     | 1-6      |       |


#### notem_hb_productions_full
The segmentation of fully segmented output of NoTEM HB Production Model. 
The lowest level of segmentation that NoTEM can handle when combined with zoning data.
Similar to `notem_hb_productions_full_tfnat`, but with the area types
removed to remove redundant 0s.

| Segment Name   | Description                                           | Values   | Notes |
|----------------|-------------------------------------------------------|----------|-------|
| `p`            | NTEM Purpose                                          | 1-8      |       |
| `tfn_tt`       | TfN Traveller Type - Extension of NTEM Traveller Type | 1-760    |       |
| `m`            | NTEM Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6 |       |
| `tp`           | NTEM Time Period.                                     | 1-6      |       |


### NoTEM HB Attractions

#### notem_lu_emp
Equivalent to the segmentation the Land Use employment data is received in.

| Segment Name | Description                                           | Values        | Notes |
|--------------|-------------------------------------------------------|---------------|-------|
| `e_cat`      | TfN Traveller Type - Extension of NTEM Traveller Type | E01, E03-E14  | E07 is split into 4. E07A-E07D. 16 values in total. |
| `soc`        | Skill Level Proxy                                     | 1-3           |      |

#### notem_hb_attractions_trip_weights
Equivalent to the segmentation the attractions trip rates come in at.
Similar to `notem_hb_attractions_pure` but with e_category too

| Segment Name | Description                                           | Values        | Notes |
|--------------|-------------------------------------------------------|---------------|-------|
| `p`          | NTEM Purpose                                          | 1-8           |       |
| `e_cat`      | TfN Traveller Type - Extension of NTEM Traveller Type | E01, E03-E14  | E07 is split into 4. E07A-E07D. 16 values in total. |
| `soc`        | Skill Level Proxy                                     | 1-3           |      |


#### notem_hb_attractions_pure
The segmentation of pure demand output of NoTEM HB Attraction Model

| Segment Name | Description                                           | Values        | Notes |
|--------------|-------------------------------------------------------|---------------|-------|
| `p`          | NTEM Purpose                                          | 1-8           |       |
| `soc`        | Skill Level Proxy                                     | 1-3           |       |


#### notem_hb_attractions_full
The segmentation of fully segmented output of NoTEM HB Attraction Model. 

| Segment Name | Description                                           | Values        | Notes |
|--------------|-------------------------------------------------------|---------------|-------|
| `p`          | NTEM Purpose                                          | 1-8           |       |
| `soc`        | Skill Level Proxy                                     | 1-3           |       |
| `m`          | NTEM Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6      |       |


### NoTEM NHB Productions

#### notem_hb_tfnat_p_m_g_soc_ns_ca
Contains all segments in the segmentation name, using the NoTEM format of
`g`/`soc`/`ns`.
The segmentation is similar to `hb_notem_output_no_tp` with TfN area type.  

| Segment Name   | Description                                           | Values   | Notes |
|----------------|-------------------------------------------------------|----------|-------|
| `tfn_at`     	 | TfN Area Type - Extension of NTEM Area Type           | 1-8      |       |
| `p`            | NTEM Purpose                                          | 1-8      |       |
| `m`            | NTEM Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6 |       |
| `g`            | NTEM Gender.                                          | 1-3      | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                     | 1-3      | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                          | 1-5      |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more              | 1-2      |       |


#### notem_nhb_trip_rate
Equivalent to the segmentation the non home based trip rate data as received in.

| Segment Name | Description                                               | Values     | Notes |
|--------------|-----------------------------------------------------------|------------|-------|
| `nhb_p`      | NTEM NHB Purpose                                          | 12-16, 18  |       |
| `nhb_m`      | NTEM NHB Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6   |       |
| `p`          | NTEM Purpose                                              | 1-8        |       |
| `m`          | NTEM Mode. Modes 3 and 4 are combined to make mode 3      | 1-3, 5-6   |       |
| `tfn_at`     | TfN Area Type - Extension of NTEM Area Type               | 1-8        |       |


#### notem_hb_tfnat_p_m_g_soc_ns_ca_nhbp_nhbm
The segmentation that results from multiplying `notem_hb_tfnat_p_m_g_soc_ns_ca`
with `notem_nhb_trip_rate`.
This segmentation is quickly aggregated into `notem_nhb_productions_pure`
within the model.

| Segment Name   | Description                                               | Values     | Notes |
|----------------|-----------------------------------------------------------|------------|-------|
| `tfn_at`     	 | TfN Area Type - Extension of NTEM Area Type               | 1-8        |       |
| `p`            | NTEM Purpose                                              | 1-8        |       |
| `m`            | NTEM Mode. Modes 3 and 4 are combined to make mode 3      | 1-3, 5-6   |       |
| `g`            | NTEM Gender.                                              | 1-3        | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                         | 1-3        | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                              | 1-5        |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more                  | 1-2        |       |
| `nhb_p`        | NTEM NHB Purpose                                          | 12-16, 18  |       |
| `nhb_m`        | NTEM NHB Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6   |       |


#### notem_nhb_productions_pure
The segmentation of pure demand output of NoTEM NHB Production Model. 

| Segment Name   | Description                                               | Values     | Notes |
|----------------|-----------------------------------------------------------|------------|-------|
| `tfn_at`     	 | TfN Area Type - Extension of NTEM Area Type               | 1-8        |       |
| `nhb_p`        | NTEM NHB Purpose                                          | 12-16, 18  |       |
| `nhb_m`        | NTEM NHB Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6   |       |
| `g`            | NTEM Gender.                                              | 1-3        | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                         | 1-3        | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                              | 1-5        |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more                  | 1-2        |       |


#### notem_nhb_productions_pure_report
A simplified reporting version of `notem_nhb_productions_pure` segmentation.

| Segment Name   | Description                                               | Values     | Notes |
|----------------|-----------------------------------------------------------|------------|-------|
| `tfn_agg_at`   | An Aggregated version of TfN Area Type                    | 1-3        | 1: 1-3, 2: 4-6, 3: 7-8  |
| `nhb_p`        | NTEM NHB Purpose                                          | 12-16, 18  |       |
| `nhb_m`        | NTEM NHB Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6   |       |
| `g`            | NTEM Gender.                                              | 1-3        | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                         | 1-3        | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                              | 1-5        |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more                  | 1-2        |       |


#### notem_nhb_tfnat_p_m_tp
Equivalent to the segmentation the non-home based time period split data as received in. 

| Segment Name   | Description                                               | Values     | Notes |
|----------------|-----------------------------------------------------------|------------|-------|
| `tfn_at`     	 | TfN Area Type - Extension of NTEM Area Type               | 1-8        |       |
| `nhb_p`        | NTEM NHB Purpose                                          | 12-16, 18  |       |
| `nhb_m`        | NTEM NHB Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6   |       |
| `tp`           | NTEM Time Period.                                         | 1-6        |       |


#### notem_nhb_productions_full_tfnat
The segmentation that results from multiplying `notem_nhb_productions_pure`
with `notem_nhb_tfnat_p_m_tp`.
This segmentation is quickly aggregated into `notem_nhb_productions_full`
within the model.

| Segment Name   | Description                                               | Values     | Notes |
|----------------|-----------------------------------------------------------|------------|-------|
| `tfn_at`     	 | TfN Area Type - Extension of NTEM Area Type               | 1-8        |       |
| `nhb_p`        | NTEM NHB Purpose                                          | 12-16, 18  |       |
| `nhb_m`        | NTEM NHB Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6   |       |
| `g`            | NTEM Gender.                                              | 1-3        | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                         | 1-3        | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                              | 1-5        |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more                  | 1-2        |       |
| `tp`           | NTEM Time Period.                                         | 1-6        |       |


#### notem_nhb_productions_full
The segmentation of fully segmented output of NoTEM NHB Production Model. 
 
| Segment Name   | Description                                               | Values     | Notes |
|----------------|-----------------------------------------------------------|------------|-------|
| `nhb_p`        | NTEM NHB Purpose                                          | 12-16, 18  |       |
| `nhb_m`        | NTEM NHB Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6   |       |
| `g`            | NTEM Gender.                                              | 1-3        | g=1 only has soc=2, and not all soc/ns combos like other genders|
| `soc`          | Skill Level Proxy                                         | 1-3        | soc=[1, 3] is not relevant for g=1                              |
| `ns`           | Household Income Level proxy                              | 1-5        |       |
| `ca`           | Car Availability. 1=No cars, 2=1 or more                  | 1-2        |       |
| `tp`           | NTEM Time Period.                                         | 1-6        |       |



### General

#### tfn_at
Segmentation containing only TfN area type.

| Segment Name | Description                                           | Values        | Notes |
|--------------|-------------------------------------------------------|---------------|-------|
| `tfn_at`     | TfN Area Type - Extension of NTEM Area Type           | 1-8           |       |


#### hb_p_m
Segmentation containing just home based NTEM purposes and modes.

| Segment Name   | Description                                           | Values   | Notes |
|----------------|-------------------------------------------------------|----------|-------|
| `p`            | NTEM Purpose                                          | 1-8      |       |
| `m`            | NTEM Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6 |       |

#### hb_p_m_6tp
Segmentation containing just home based NTEM purposes, modes, and 6 time periods.

| Segment Name   | Description                                           | Values   | Notes |
|----------------|-------------------------------------------------------|----------|-------|
| `p`            | NTEM Purpose                                          | 1-8      |       |
| `m`            | NTEM Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6 |       |
| `tp`           | NTEM Time Period.                                     | 1-6      |       |

#### nhb_p_m_6tp
Segmentation containing just home based NTEM purposes, modes, and 6 time periods.

| Segment Name   | Description                                           | Values   | Notes |
|----------------|-------------------------------------------------------|----------|-------|
| `p`            | NTEM Purpose                                          | 12-16, 18|       |
| `m`            | NTEM Mode. Modes 3 and 4 are combined to make mode 3  | 1-3, 5-6 |       |
| `tp`           | NTEM Time Period.                                     | 1-6      |       |
