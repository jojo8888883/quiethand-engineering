# QuietHand 30 条 evaluation 质检盲测 Result-to-Claim

- `claim_supported: partial`
- `confidence: high`
- `integrity_status: pass_with_limitations`
- `review_independence: same-family`
- `acceptance_status: provisional`

## 最窄成立主张

在冻结的 30 条 evaluation 上，以一次 Codex 盲视审阅产生的 proxy 标签为准、同样复核 6 条时，几何增强排序抓到 2 / 4 个核心粗标错误，而 VLM-only（语义不确定性全为 0、按 opaque ID 打破并列）抓到 1 / 4 个，未复核误放相应由 3 个降至 2 个。

## 直接支持什么

- 排名、目标和计分可复算，预算同为 6 / 30，没有目标泄漏、训练或结果后调参。
- 在当前 proxy 目标上，geometry 相比冻结 comparator 的 recall@6 为 `0.50 vs 0.25`，precision@6 为 `0.333 vs 0.167`，多抓到 1 个错误。
- 支持的是 event-level 质检优先级排序；固定错误总数和复核预算下，少误放 1 个是多抓到 1 个的同一结果，不是第二项独立收益。

## 不支持什么

- 不能把这 4 个判断称为独立人工 gold 下确认的真实错误率。
- 不能声称几何已经优于一个有判别力的 VLM 风险模型；本次 VLM-only 全并列，实际 comparator 是冻结的 opaque-ID 顺序。
- 不支持字段级自动纠错、跨样本或跨数据集泛化、训练收益、标注降本、下游策略或机器人收益。

## 真正缺的下一份证据

若以后要把该信号升级成一般方法主张，需要一批预先冻结的新 held-out 数据、独立于当前 agent 的人类目标，以及一个分数不退化的 semantic-only baseline，在相同复核预算下比较。该工作没有被本轮自动授权。

