# evn - demo

```
conda create -n LTAP 3.8.11   
source activate LTAP 
pip install -r requirements.txt
python main.py --save_all_epoch --hesso --target_group_sparsity 0.1 --isc LT-vote --imb_ratio 100  --bs --gpu 1 > bs_1.log  2>&1 &
```

All the command templates have been placed in ``1.sh``; you can run it directly or execute individual commands from it.

# Additional 
### Ablation Study Analysis on CIFAR-100-LT (ir=100)
We conduct an ablation study analysis on the CIFAR-100-LT dataset with an imbalance ratio (ir) of 100. Specifically, "Base" refers to the baseline method for long-tailed tasks, while "ours w.o. PSL" denotes the scenario where $\alpha_k \equiv 1$. 

<img src="https://anonymous.4open.science/r/AEFCDAISJ/anix.png" alt="示例图片" style="width: 300px; margin: 10px; border: 1px solid #ccc;">

### Pseudocode
<img src="https://anonymous.4open.science/r/AEFCDAISJ/algorithmic.png" alt="示例图片" style="width: 300px; margin: 10px; border: 1px solid #ccc;">

### ir =50
| Method   | F    | Head  | Medium | Tail  | All   | C     | C/F |
|--------|------|-------|--------|-------|-------|-------|-----|
| ce     | 100.0| 68.0  | 38.0   | 13.2  | 46.0  | 100.0 | 1.0 |
| ce + ato | 84.7 | 46.7  | 17.3   | 6.83  | 29.1  | 63.2  | 0.74|
| ce + regg | 52.1 | 43.8  | 14.8   | 0.83  | 24.5  | 53.2  | 1.02|
| ce + ours | 23.3 | 64.8  | 31.7   | 7.2   | 41.1  | 89.3  | 3.83|
| la     | 100.0| 59.9  | 46.7   | 41.3  | 51.3  | 100.0 | 1.0 |
| la + ato | 84.7 | 34.5  | 33.9   | 29.1  | 34.2  | 66.7  | 0.79|
| la + regg | 52.1 | 31.0  | 30.2   | 25.8  | 29.9  | 58.3  | 1.12|
| la + ours | 22.8 | 54.0  | 43.4   | 38.4  | 41.2  | 80.3  | 3.52|
### ir = 100
| Method   | F    | Head  | Medium | Tail  | All   | C     | C/F   |
|------------|------|-------|--------|-------|-------|-------|-------|
| ce         | 100.0| 70.7  | 40.0   | 7.2   | 41.0  | 100.0 | 1.0   |
| ce + ato   | 84.7 | 50.4  | 16.5   | 6.6   | 25.2  | 61.5  | 0.73  |
| ce + regg  | 52.1 | 47.7  | 13.6   | 0.6   | 21.9  | 53.4  | 1.02  |
| ce + ours  | 23.3 | 66.1  | 31.7   | 2.5   | 35.1  | 85.6  | 3.67  |
| la         | 100.0| 62.9  | 47.7   | 29.6  | 47.9  | 100.0 | 1.0   |
| la + ato   | 84.7 | 42.1  | 30.6   | 18.4  | 31.4  | 65.6  | 0.77  |
| la + regg  | 52.1 | 38.5  | 27.0   | 14.6  | 27.6  | 57.6  | 1.11  |
| la + ours  | 22.8 | 56.1  | 45.3   | 22.6  | 42.6  | 88.9  | 3.90  |

