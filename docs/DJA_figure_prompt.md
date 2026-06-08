# DJA Methodology Figure — Design Spec & Drawing Prompt

## 设计理念

图的核心叙事：**DJA 不是"朝固定目标的静态优化"，而是"不断适应模型当前分布的动态闭环搜索"。**

三个 Dynamic 机制是同一动态哲学的三种表达，不是三个独立 trick。
Teal 色（#2C9F8F）作为"动态"的专属颜色，贯穿全图所有动态元素。


## 完整绘图 Prompt

```
Create a publication-quality methodology figure for a machine learning security
paper on a pure white background. Flat design, IEEE/NeurIPS style, clean 
sans-serif font (Inter or Helvetica), no 3D effects, no shadows, vector-art 
quality. Aspect ratio 16:9. 

═══════════════════════════════════════════════════
FIGURE TITLE (centered at top)
═══════════════════════════════════════════════════
Bold title: "Dynamic Jailbreaking Attack (DJA)"
Subtitle below in gray italic: "Adaptive Closed-Loop Search vs. Static Paradigm"

═══════════════════════════════════════════════════
LEFT PANEL — "Static Baseline" (25% of width)
═══════════════════════════════════════════════════
A narrow vertical panel with a light gray (#F5F5F5) background, 
labeled at top in gray: "Existing (Static)"

Inside, three stacked items each with a RED X icon on the right:
  ✗  Fixed target response  r_fixed
     (small box, dashed gray border)
  ✗  Fixed sampling budget  n
     (bar showing one fixed-length bar)
  ✗  Fixed suffix length  L
     (ruler icon with fixed length)

Below the three items, a gray annotation:
  "Target: externally preset template
   → low-probability region of model distribution"

A vertical dashed divider separates this panel from the right.
Above the divider, a horizontal arrow pointing RIGHT labeled in teal:
  "DJA: make all three dynamic →"

═══════════════════════════════════════════════════
RIGHT PANEL — "DJA Dynamic System" (75% of width)
═══════════════════════════════════════════════════
This panel has a subtle light blue tint (#F0F7FF) background.

Inside this panel is the MAIN LOOP — visualized as a large OVAL/CIRCULAR 
flow path (not a rectangular pipeline). The oval loop is drawn with a 
THICK TEAL ARROW (#2C9F8F, 3px stroke) going CLOCKWISE, with the label 
on the loop path: "adapts every outer iteration t → t+1"
This oval loop is the dominant visual element of the figure.

──────────────────────────────────────
POSITION: LEFT of the oval (loop entry point)
──────────────────────────────────────
Input stacked boxes:
  Box 1 (gray fill): "Harmful Prompt  p"
  Box 2 (warm orange gradient fill #FEEBC8→#F6AD55): 
        "Adversarial Suffix  s_t"
        Inside: 8 small colored squares showing soft token logits
        Subscript below: "iteration t"

──────────────────────────────────────
POSITION: TOP of the oval (12 o'clock)
──────────────────────────────────────
A hexagon, light blue fill (#EBF8FF), dark blue border (#2B6CB0).
Label: "Target LLM  f_θ"
       "(white-box, gradient access)" in smaller italic below
An open padlock icon inside.

From hexagon, TWO outgoing paths:

PATH A (going RIGHT, for sampling):
Arrow labeled "high-temp sampling  T = 2.0"
Fans out into a vertical column of candidate boxes:
  r₁  [gray box]
  r₂  [gray box]  
  r₃  [gray box]
  ⋮
  rₙ  [gray box, with "..." below]

A small teal expansion bracket to the right of the column:
  "↕ n doubles if no unsafe found
   n → 2n → ··· → n_max"
  (this is the DYNAMIC BUDGET visual)

PATH B (going DOWN-LEFT, for gradient, part of inner loop):
Described below.

──────────────────────────────────────
POSITION: RIGHT of the oval (3 o'clock)
──────────────────────────────────────
MULTI-OBJECTIVE SCORER box (light purple fill #FAF5FF, purple border):
Label: "Multi-Objective Scorer"
Inside: 5 horizontal score bars (proportional lengths):
  Harm          ████████████████  0.70
  Specificity   ███               0.09
  Relevance     ███               0.09
  Coherence     ██                0.06
  Non-refusal   ██                0.06

A small filter funnel icon with "Degeneracy Gate" label below the scorer,
showing responses being filtered before scoring.

Below scorer, one highlighted green box (light green fill #F0FFF4, 
green border #276749, slightly larger than others):
  "Best Response  r*_t"

Key visual: Show a SMALL INSET COMPARISON beside r*_t:
  Two mini-boxes side by side:
    [Iter t:   r*_t  = "Here are the steps to..."]  
    [Iter t+1: r*_{t+1} = "First, you need to..."]
  Connected by a right arrow, with caption: "target shifts each iteration"
  (use blurred/gray placeholder text, no real harmful content)
  This inset visually proves the "dynamic" claim.

──────────────────────────────────────
POSITION: BOTTOM of the oval (6 o'clock) — INNER LOOP
──────────────────────────────────────
A smaller nested dashed-border region (orange dashed, #C05621),
labeled "Inner Loop  (×num_inner_iters)" in orange at top.

Inside, three boxes connected by downward arrows:
  [CE Loss: -log P(r*_t | p ⊕ s_t)]
          ↓
  [∇ gradient w.r.t. suffix logits]
          ↓
  [Update s via AdamW]

To the right of this inner loop box, a small visual:
  A growing bar chart showing suffix length:
  [████] → [████████] → [████████████]
  Caption: "suffix length expands on plateau"
  (this is the DYNAMIC SUFFIX visual)

──────────────────────────────────────
THE TEAL OVAL ARROW completes the loop:
From "Update s via AdamW" → curves LEFT and UP → 
back to Input "Adversarial Suffix  s_{t+1}"
The subscript change  s_t → s_{t+1}  on the arrow
shows temporal progression explicitly.

──────────────────────────────────────
EXIT from loop (when success):
A downward arrow exits the bottom-right of the oval, 
labeled "composite score ≥ θ  →  attack succeeds"
Points to a final box (red border, light pink fill):
  "Jailbroken Response"
  [two lines of blurred gray bars — no actual text]

═══════════════════════════════════════════════════
UNIFIED "DJA DYNAMIC MECHANISMS" LEGEND BOX
(bottom-right corner of the right panel)
═══════════════════════════════════════════════════
A single rounded rectangle (teal border #2C9F8F, very light teal fill)
labeled at top: "DJA: Three Dynamic Mechanisms"

Inside, THREE rows, each with a teal circle number and short label:
  ① Dynamic Target Tracking
    "r* re-selected from current  P(r | p⊕s_t)  each outer iter"

  ② Dynamic Sampling Budget  
    "budget expands until unsafe response is found"

  ③ Dynamic Suffix Capacity
    "suffix length grows when optimization plateaus"

Below all three rows, a unifying italic line in teal:
  "All three adapt continuously — no static configuration"

(This box REPLACES scattered badges. All three mechanisms share ONE
 visual language: teal color, one box. Communicates: three expressions
 of ONE dynamic philosophy.)

═══════════════════════════════════════════════════
COLOR SCHEME
═══════════════════════════════════════════════════
Main loop arrow:        #2C9F8F  (teal — the "dynamic" color)
Dynamic legend box:     border #2C9F8F, fill #E6FFFA
Target LLM hexagon:     fill #EBF8FF, border #2B6CB0
Suffix box:             gradient #FEEBC8 → #F6AD55
Best response r*:       fill #F0FFF4, border #276749
Multi-objective scorer: fill #FAF5FF, border #6B46C1
Static panel:           fill #F5F5F5, all elements in gray, red X marks
Inner loop border:      #C05621 dashed
Background (DJA panel): #F0F7FF very light blue tint
Background (static):    #F5F5F5 gray tint

KEY RULE: Teal (#2C9F8F) appears ONLY on dynamic elements.
It threads through: the main loop arrow, the legend box, the budget
expansion bracket, and the suffix growth visual. This creates a visual
system where "teal = dynamic adaptation."

═══════════════════════════════════════════════════
STYLE KEYWORDS
═══════════════════════════════════════════════════
academic diagram, flat vector design, publication quality, 
clean white and teal color scheme, NeurIPS paper figure,
technical flow diagram, no clip art, no gradients except suffix box,
precise label placement, mathematical notation in italic
```
