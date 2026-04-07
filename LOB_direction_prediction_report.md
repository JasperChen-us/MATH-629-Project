# Handling the “Always Flat” Problem in High-Frequency LOB Direction Prediction

## Overview

In high-frequency limit order book (LOB) direction prediction, a common issue is that the model learns to predict **flat / no-change** most of the time. This can happen because the flat class is often the majority class over short horizons. As a result, the model may achieve strong **raw accuracy** while producing a signal that is economically useless.

This report explains why this happens, why simply predicting flat is not a good solution, and what modifications should be made to the **labels, loss function, sampling scheme, and evaluation metrics** to produce a more meaningful and tradable model.

---

## 1. Why “always flat” can look optimal

In high-frequency data, the mid-price frequently remains unchanged over very short horizons. If we define a 3-class target:

- **Up**
- **Flat**
- **Down**

then the flat class often dominates the dataset. In that case, a model that predicts **flat** most of the time can achieve:

- high plain accuracy
- low apparent classification error
- strong benchmark performance under naive metrics

However, this does **not** mean the model has learned useful market structure. It may simply be exploiting label imbalance.

The problem is especially severe when labels are defined from a **single future timestamp**, because tiny microstructure noise can dominate the target.

---

## 2. Why this is a bad solution in practice

Although predicting flat can be statistically safe, it is not a good trading solution because:

- it does not identify **when a meaningful move will occur**
- it provides little or no **actionable trading signal**
- it can fail badly once evaluation shifts from raw accuracy to **macro metrics** or **PnL-style metrics**
- it encourages the model to ignore rare but important **up/down events**

So the goal is **not** to force the model to predict non-flat artificially.  
The goal is to redesign the prediction task so that:

1. flat does not dominate trivially,
2. economically meaningful moves are emphasized,
3. the evaluation clearly exposes useless “always flat” behavior.

---

## 3. Improve the label definition first

A major improvement is to define labels using a **smoothed future price** rather than a single future mid-price snapshot.

A common approach is:

\[
a^+(k,t)=\frac{1}{k}\sum_{i=1}^{k} m(t+i)
\]

where:

- \(m(t)\) is the current mid-price
- \(a^+(k,t)\) is the average future mid-price over the next \(k\) steps

Then define the label using a threshold \(\theta\):

\[
y_t =
\begin{cases}
1, & a^+(k,t) > m(t)(1+\theta) \\
0, & m(t)(1-\theta) \le a^+(k,t) \le m(t)(1+\theta) \\
-1, & a^+(k,t) < m(t)(1-\theta)
\end{cases}
\]

This improves the task because:

- it reduces sensitivity to micro-noise
- it prevents tiny meaningless fluctuations from becoming up/down labels
- it makes labels more aligned with short-term trend rather than one-tick randomness

### Practical label improvements

- use **future averaged mid-price** instead of one future timestamp
- introduce a **nonzero threshold**
- choose threshold based on:
  - tick size
  - spread
  - short-term volatility
  - trading cost considerations

---

## 4. Punish the flat solution through the loss

The most direct way to reduce “always flat” collapse is to use a **class-weighted loss**.

### Weighted cross-entropy

For class label \(y\), use:

\[
\mathcal{L} = - w_y \log p_y
\]

where:

- \(p_y\) is the predicted probability of the true class
- \(w_y\) is a class-dependent weight

Set:

- higher weight for **up**
- higher weight for **down**
- lower weight for **flat**

A common choice is:

- inverse-frequency weights
- inverse-square-root-frequency weights

Example:

- if class proportions are approximately `down : flat : up = 1 : 8 : 1`
- then weights might be approximately `8 : 1 : 8`

### Why this helps

This penalizes the model more heavily when it misses rare move classes.  
Without this, the model can minimize loss by focusing on flat.

### Stable practical choice

Inverse-frequency can sometimes be too aggressive.  
A safer option is:

\[
w_c \propto \frac{1}{\sqrt{n_c}}
\]

where \(n_c\) is the number of samples in class \(c\).

---

## 5. Try focal loss, but do not assume it is always best

**Focal loss** is another option for imbalanced classification. It downweights easy examples and puts more emphasis on hard examples.

This can help when the model is overconfident on flat examples.

However, focal loss is **not guaranteed** to outperform weighted cross-entropy in high-frequency LOB tasks. In practice:

- it is worth testing
- but weighted CE or cost-sensitive CE may still work better

So focal loss should be treated as an experiment, not an automatic replacement.

---

## 6. Rebalance the training data

Another way to reduce flat domination is to rebalance the training dataset.

### Good options

- **Downsample flat** in the training set
- use a **WeightedRandomSampler**
- combine sampling with weighted loss

### Important rule

Only rebalance the **training set**.

Do **not** rebalance validation or test sets, because:

- you want evaluation to reflect the true market distribution
- otherwise results become unrealistic

---

## 7. Use a two-stage modeling design

A strong structural improvement is to split the task into two stages.

### Stage 1: event detection
Predict:

- **move**
- **no move**

### Stage 2: direction prediction
Conditional on “move”, predict:

- **up**
- **down**

### Why this helps

This separates two different problems:

1. **Is there anything worth trading?**
2. **If yes, which direction is it?**

This is often better than forcing a single 3-class classifier to solve:

- up
- flat
- down

all at once.

### Trading interpretation

This is closer to real strategy logic:

- first detect whether a meaningful price change is likely
- then decide the side of the trade

---

## 8. Consider removing flat entirely in one version of the task

Another strong alternative is to only keep samples with meaningful future movement.

For example, only keep timestamps satisfying:

\[
|a^+(k,t)-m(t)| > \theta m(t)
\]

Then:

- either build a **binary up/down classifier**
- or build a **tradeable move vs no-trade** classifier

This can be especially useful if your eventual strategy only trades when the expected move is large enough to overcome:

- spread
- fees
- slippage

---

## 9. Use evaluation metrics that expose the flat-collapse problem

Do **not** rely only on raw accuracy.

If flat is 80% of the samples, an “always flat” model can score around 80% accuracy and still be useless.

### Better classification metrics

Track:

- **Macro F1**
- **Balanced Accuracy**
- **Macro Precision**
- **Macro Recall**
- **Per-class precision / recall / F1**
- **Confusion matrix**

These metrics treat minority classes more fairly and make flat-collapse visible.

### Useful additional diagnostics

Also track:

- recall on **up/down only**
- precision of non-flat predictions
- proportion of predicted non-flat events
- hit rate conditional on emitting a trade signal

### Trading-oriented evaluation

If possible, also evaluate:

- simple **PnL after spread and fees**
- average return of predicted up/down trades
- precision at high-confidence thresholds
- trade frequency vs profitability

This is important because the final goal is not just classification quality, but **economic usefulness**.

---

## 10. Recommended modifications to the pipeline

## A. Dataset / label changes

Replace raw next-mid label with a smoothed, thresholded label.

### Recommended target

1. Compute current mid-price:
\[
m(t) = \frac{best\ bid_t + best\ ask_t}{2}
\]

2. Compute future average:
\[
a^+(k,t)=\frac{1}{k}\sum_{i=1}^{k} m(t+i)
\]

3. Define label with threshold \(\theta\):
- `up` if future average is sufficiently above current mid
- `down` if sufficiently below
- `flat` otherwise

### Notes

- choose \(k\) based on your horizon
- choose \(\theta\) based on tick size / spread / volatility
- log the resulting class distribution before training

---

## B. Loss changes

Start with **weighted cross-entropy**.

### Recommended order of experiments

1. ordinary cross-entropy
2. weighted cross-entropy
3. focal loss
4. custom cost-sensitive loss

### Suggested first choice

Weighted CE with inverse-sqrt-frequency weights.

---

## C. Sampling changes

In the training set only:

- use **WeightedRandomSampler**
- or moderate flat downsampling

This helps prevent the network from seeing flat examples overwhelmingly more often than move examples.

---

## D. Model-design changes

Try both:

### Option 1: 3-class direct model
- up / flat / down

### Option 2: two-stage model
- move vs no-move
- up vs down given move

The second design is often more aligned with execution logic.

---

## E. Evaluation changes

Replace single-metric evaluation with:

- macro F1
- balanced accuracy
- per-class recall
- confusion matrix
- non-flat precision / recall
- simple cost-aware backtest

A model should not be considered good if it only performs well under plain accuracy.

---

## 11. Practical recommendations for your project

If this were my LOB pipeline, I would do the following:

### Step 1
Redefine labels using:

- averaged future mid-price
- threshold-based up / flat / down classification

### Step 2
Use **weighted cross-entropy** with weights computed from the training set.

### Step 3
Apply **WeightedRandomSampler** or mild flat downsampling in the training loader.

### Step 4
Evaluate using:

- macro F1
- balanced accuracy
- confusion matrix
- non-flat recall
- simple cost-aware trading metrics

### Step 5
Build a second version of the model using a **two-stage structure**:
- move vs no-move
- up vs down

### Step 6
Compare against strong baselines:
- always flat
- majority class
- zero-return / no-change
- simple order imbalance signal

---

## 12. Key takeaway

The solution is **not** to manually force the model away from flat predictions in an arbitrary way.  
The correct solution is to redesign the task so that:

- labels represent meaningful moves
- the loss penalizes ignoring rare move classes
- the training sampler is not dominated by flat cases
- the evaluation makes useless flat-prediction strategies fail clearly

In short:

- **better labels**
- **weighted loss**
- **rebalanced training**
- **better metrics**
- **event-based or two-stage design**

These changes make the model much more likely to learn something economically meaningful rather than just exploiting label imbalance.

---

## 13. Short summary of modifications

### Old problem
- 3-class direction prediction
- flat is the majority class
- model can learn “always flat”
- plain accuracy looks strong, but the signal is useless

### New solution
- smooth the future target using future average mid-price
- add a threshold to avoid labeling tiny moves
- use weighted cross-entropy
- optionally test focal loss
- rebalance the training data
- evaluate with macro metrics, not just accuracy
- consider a two-stage model
- optionally remove flat entirely for a tradeable-event task

---

## 14. Suggested implementation checklist

- [ ] compute current mid-price
- [ ] compute future averaged mid-price over horizon \(k\)
- [ ] choose threshold \(\theta\)
- [ ] generate up / flat / down labels
- [ ] inspect class imbalance
- [ ] compute class weights from training set
- [ ] train with weighted cross-entropy
- [ ] test WeightedRandomSampler
- [ ] compare with focal loss
- [ ] evaluate with macro F1 and balanced accuracy
- [ ] inspect confusion matrix
- [ ] backtest non-flat predictions with costs
- [ ] test two-stage model
- [ ] compare with always-flat and simple baselines

---

## 15. Reference directions for literature review

When extending this report, useful literature directions include:

- deep learning for LOB forecasting
- mid-price trend labeling with future average price
- class imbalance learning in HFT
- cost-sensitive classification
- focal loss under imbalanced labels
- event-based labeling and tradeable signal design
- macro-F1 / balanced-accuracy evaluation under class imbalance

These are the key themes behind the modifications recommended above.
