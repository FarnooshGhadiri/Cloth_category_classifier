# Cloth_category_classifier

The goal of our approach is to have a fully automatic system to detect COs on any frame where a person appears in the camera field of view. Using only one frame to detect COs makes the algorithm robust to events such as leaving the scene, handling over a luggage, or a change of direction of the person. To detect carried objects, we build on two sources of information. The first is the output of the person’s contours hypothesis generator. The second source of information is the output of a bottom-up object segmentation. Our contribution is to combine this information to discriminate between carried object contours and other objects (person, background).

![img](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/attributes.jpg)

For each tracked person, a person’s contour hypothesis mask is generated based on the similarity of its contour to the sets of contours obtained from humans in different views. Contours with low scoring person’s contour hypothesis are considered as hypothesized CO contours. Each contour candidate of CO is 
closed and the obtained enclosed region is uniformly sampled. A Biased Normalized Cut is used to assign a region to each set of samples. Prior knowledge of person’s contour hypothesis and a segmented foreground is used to eliminate false positive regions.



## Troubleshooting

If you are experiencing any issues, please file a ticket in the Issues section.
