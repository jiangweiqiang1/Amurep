# Amurep model
In this paper, a method Amurep(Adaptively Multi-layer Region Partitioning) is presented, for adaptively aggregating POIs into multi-layer POI groups to improves recommendation performance. Specifically, POI groups in multi-scales are learned to constrain the range of recommended next POI. In addition, for better capturing users’ dynamic requirements, high-order information, such as multi-hop relations of POIs within user check-in trajectories, are used. In the end, extensive experiments are conducted on two real-world datasets to show that Amurep outperforms the state-of-the-art methods to the best of our knowledge and the learned POI groups are also interpretable.
# Environment
numpy==1.19.2  
tqdm==4.58.0  
pandas==1.1.5  
scipy==1.6.1  
torch_summary==1.4.5  
networkx==2.5  
torchsummary==1.5.1  
# Data
The data is from two global datasets, available at websites https://snap.stanford.edu/data/loc-gowalla.html（gowalla dataset） and https://sites.google.com/site/yangdingqi/home（gowalla dataset）. For detailed preprocessing steps, refer to the data preprocessing section of the paper.
# Training
python main.py
# Citation
If you use our code, please kindly cite:  
<div style="overflow: hidden; width: 100%; position: relative;">
  <div style="display: flex; animation: slide 10s infinite;">
    <div style="flex: 1; text-align: center; padding: 20px;">
      <pre>
@inproceedings{jiang2025amurep,
  title = {Amurep: Adaptively Multi-layer Region Partitioning for Next POI Recommendation},
  author = {Jiang, Weiqiang and Wang, Yan and Liu, Lijuan and Zhu, Shunzhi},
  booktitle = {Proceedings of the 2025 International Joint Conference on Neural Networks (IJCNN)},
  year = {2025}
}
      </pre>
    </div>
    <div style="flex: 1; text-align: center; padding: 20px;">
      <!-- You can place another reference or text here if needed -->
      <p>Another sliding content here if needed.</p>
    </div>
  </div>
</div>

<style>
  @keyframes slide {
    0% { transform: translateX(0); }
    100% { transform: translateX(-100%); }
  }
</style>


