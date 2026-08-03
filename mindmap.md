# NHC0801 科学流水线 Mindmap

> 步骤 **0 → 12** 顺序执行；**Train / Validation / Final Test** 按 molecular root 隔离。

---

## 0. 冻结全部原始分子

每个 molecular root 包含：

- NHC-H+ 阳离子
- NHC 中性体

**冻结：** 原子顺序、初始坐标、charge、multiplicity、结构 SHA256

---

## 1. 按 molecular root 划分

| 集合 | 示意 |
| --- | --- |
| Train roots | A–E |
| Validation roots | F–G |
| Final Test roots | H–I |

三组必须完全分开：

- Train ∩ Validation ∩ Test = 空集
- 同一个分子的阳离子、中性体、轨迹和所有后代结构不能拆开

---

## 2. 用 Pure PySCF 生成老师答案

每个 root 的阳离子和中性体分别执行：

1. 冻结原始几何  
2. 直接 parent-level PySCF/geomeTRIC 完整优化  
3. 最终达到 parent-level GAU  
4. 保存每一步：
   - 几何坐标
   - PySCF 能量
   - PySCF 原子梯度/力
   - charge / multiplicity
   - lineage / protocol SHA

输出去向：

| 数据 | 用途 |
| --- | --- |
| Train 数据 | 可用于训练 |
| Validation 参考 | 不用于训练 |
| Test 参考 | 完全封存 |

---

## 3. 训练前 Epoch-0 基线

使用官方未微调 AIMNet2：

1. Validation 原始几何  
2. 官方 AIMNet2（epoch-0）  
3. AIMNet2 自身达到 GAU_LOOSE  
4. exact-byte handoff  
5. PySCF/geomeTRIC 完整优化到最终 GAU  
6. 与 Pure-PySCF reference 比较  

得到「微调前」的正式基线。

---

## 4. 正式训练 AIMNet2

只允许读取 Train 中保存的 PySCF frames。

1. Train frame  
2. AIMNet2 预测 energy 和 forces  
3. 与 PySCF 真实 energy 和 forces 比较  
4. 计算 training loss  
5. 反向传播，更新 AIMNet2 参数  

一个 epoch = 全部 Train frames 学习一遍。

---

## 5. 训练多个 Epoch 并保存 Checkpoint

```text
官方模型
  → epoch 1
  → epoch 2
  → epoch 3
  → …
  → epoch 100
```

每隔若干 epoch 保存 checkpoint。

---

## 6. 训练中的快速 Validation

使用 Train 之外的 Validation PySCF frames。

1. 当前 checkpoint  
2. 预测固定 Validation frame 的 energy 和 forces  
3. 与保存的 PySCF 标签比较  
4. 得到 Validation loss  

注意：

- 不运行新的 PySCF  
- 不反向传播  
- 不更新模型参数  
- 用来发现过拟合和筛选 checkpoint  

---

## 7. 初步筛选少数候选 Checkpoint

例如从很多 epoch 中选出：

- epoch 20：较早的优秀模型  
- epoch 35：Validation loss 最低  
- epoch 50：训练中期模型  
- epoch 100：最后模型  

这里只是快速筛选，还不是最终模型选择。

---

## 8. 完整科学 Validation

对少数候选 checkpoint 分别执行完整路线：

1. 同一个 Validation 冻结原始几何  
2. 候选 AIMNet2 checkpoint  
3. AIMNet2 自身达到 GAU_LOOSE  
4. exact-byte handoff  
5. PySCF/geomeTRIC 完整优化到最终 GAU  
6. 计算最终 parent-level 能量和脱质子标签  

---

## 9. Validation 负责最终模型选择

比较三类路线：

| | 路线 |
| --- | --- |
| A | Pure-PySCF reference |
| B | epoch-0 AIMNet2 → PySCF |
| C | fine-tuned checkpoint → PySCF |

选择顺序：

1. 结构、拓扑、charge、multiplicity 全部正确  
2. 没有碎裂或错误成键  
3. 最终 parent-level 标签保持  
4. 相比 epoch-0 不退化  
5. PySCF 优化步数减少  
6. SCF cycles 减少  
7. 总 wall time 减少  

最终选出一个 checkpoint。

---

## 10. 全部内容正式冻结

冻结：

- 最终 checkpoint SHA256  
- Train split  
- Validation split  
- Final Test split  
- GAU_LOOSE 合同  
- PySCF parent protocol  
- label error 阈值  
- checkpoint 选择规则  
- 代码 commit / runtime  

---

## 11. Final Test

使用模型从未训练、也未用于选择的 Test 分子：

1. Test 冻结原始几何  
2. 已冻结的最终 AIMNet2 checkpoint  
3. AIMNet2 自身达到 GAU_LOOSE  
4. exact-byte handoff  
5. PySCF/geomeTRIC 完整优化到最终 GAU  
6. 与 sealed Pure-PySCF reference 比较  
7. 输出一次性最终结果  

---

## 12. Test 后不允许继续选择

Test result出来后禁止：

- ✗ 更换checkpoint  
- ✗ 修改规则  
- ✗ 继续训练  
- ✗ 删除失败分子  
- ✗ 把Test分子加入Train后重新考试  

否则原Test必须降级为开发数据。
