# Description

This example consists of constructing walls using three types of stones: regular (R), partially regular (PR) and irregular (IR). The stone wall images are labeled from photos in [Almeida et al. 2021](https://www.sciencedirect.com/science/article/pii/S0950061821030804).The three example can be executed together by executing _python run_all.py_ script.

The configurations in wall construction are in the _config.json_ file in each subfolder. The parameters to be defined are :

- input_wall_name: the name of the image file in data/ that contains the stone set.
- scale: the size reduction ratio of the input image. Smaller ratio lead to smaller images and faster computation.
- wall_width_plus: padding to be added to the original wall image width to form the desired wall size.
- wall_height_plus:padding to be added to the original wall image height to form the desired wall size.,
- wall_width_times: the final wall width is wall_width_times\*origional_wall_width+wall_width_plus
- wall_height_times: the final wall width is wall_height_times\*origional_wall_height+wall_height_plus
- vendor_type: different strategies to sample stones from dataset. Options are: "sample_twice", "variant", "full".
- construction_order: construction strategies. Options are: "two_sides","from_left","from_right".
- variant_sample: number of candidate stones to be sampled in each step if vendor_type is "variant".
- allow_cutting: whether changing stone shapes is allowed. set to "false" for a stable performance.
