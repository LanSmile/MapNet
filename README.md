# MapNet: Manifold Augmented Positive-incentive Noise Network for Infrared Small Target Segmentation
## Overall Framework
![image name](https://github.com/LanSmile/MapNet/blob/main/Fig/overall_architecture.png)
Overall architecture of MapNet. The proposed framework consists of two collaborative branches, namely Map-gen and Map-seg.
 
## Main Contributions
1. We propose MapNet, which generates $\pi$-noise along the data manifold to enhance discriminative cues of target regions while preserving the intrinsic manifold geometry, thereby achieving robust target–background separation.
2. We design the generator module Map-gen, which models the conditional noise distribution through a learnable generator. By maximizing the variational lower bound of mutual information, it enables the amplitude and spatial distribution of the generated noise to be adaptively adjusted during end-to-end training.
3. We design the segmentation module Map-Seg. Through a shared feature extractor, noise generation and segmentation are jointly optimized under geometric constraints, ensuring sustained positive-incentive effects of the generated noise.


