# Stones

data/stones*noise*(10-80)\_labels.png: Binary wall images labeled from images in [Thangavelu et al.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460562)

# Wall construction

00_run.py: Generate 20 different layouts for each stone set. The possible stone poses range from 0-360 degrees with 90 degree interval. The input stone images are scaled by 0.5. Wall height is 5 times the average bounding box height of all regions larger than 50. Wall width is 5 times the average width of that. Built with Variant method.
01_run_with_10_interval.py: The possible stone poses range from 0-360 degrees with 10 degree interval.
