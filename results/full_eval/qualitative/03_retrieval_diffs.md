# Paired Retrieval Diffs — Helping vs Hurting

> **TL;DR.** Helping cases (original→gold\_redundant, F1+>0.1): A=464, B=359 questions. In both pipelines, gold twins displace predominantly noise (non-gold items), with mean gold\_displaced≈0.06 (A) / 0.06 (B). Hurting cases (original→negative\_redundant, F1−>0.1): A=223, B=243 questions. Negative twins displace mean 0.19 (A) / 0.21 (B) gold items per question.

---

## Pipeline A

### Helping cases  (original → gold\_redundant, F1 Δ > +0.1)

#### Aggregate statistics

| Metric | Value |
|---|---|
| n\_cases | 464 |
| avg twins added to top-10 | 1.384 |
| avg gold items displaced | 0.065 |
| avg noise (non-gold) displaced | 1.401 |
| avg Δ F1 | +0.2560 |

#### Representative examples (5)

##### q\_id=323 · Analytical · table|text

**Question:** Which model achieves the highest F1-value for Named Entity Recognition (NER) and what features contribute to it?

**gold\_set:** ['image3', 'image4', 'text9']

**Δ F1:** 0.215 → 0.519 (+0.303)

**Items dropped from top-10:**

- `text10` [TEXT, "There exist several open-source statistical CRF sequence labeling toolkits, such…"]
- `text7` [TEXT, "Table  1  shows the results of six CRF-based mod- els with different character s…"]

**Items added to top-10:**

- `image3_text` [TWIN, src=image3]
- `image4_text` [TWIN, src=image4]

| | original | gold_redundant |
|---|---|---|
| prediction | The model achieving the highest F1-value for Named Entity Recognition (NER) is the one built on the $\mathrm{NCRF++}$ toolkit, which utilizes both character sequence and word sequence representations. This model incorporates both handcrafted features… | The model that achieves the highest F1-value for Named Entity Recognition (NER) is CCNN+WLSTM+CRF, with an F1-value of 91.35. This model uses the combination of CCNN, WLSTM, and CRF layers, which contributes to its superior performance in NER tasks. |

##### q\_id=1995 · Application-based · table

**Question:** How do the different metering modes and focus modes in Pro Mode enhance photography under various scenarios?

**gold\_set:** ['image4', 'image5']

**Δ F1:** 0.173 → 0.358 (+0.184)

**Items dropped from top-10:**

- `text7` [TEXT, "• Adjust the ISO sensitivity: Touch  ISO  and drag the slider. In low-light sett…"]
- `text8` [TEXT, "• Adjust the metering mode: Touch  M  and select a metering mode."]

**Items added to top-10:**

- `image4_text` [TWIN, src=image4]
- `image5_text` [TWIN, src=image5]

| | original | gold_redundant |
|---|---|---|
| prediction | The different metering modes and focus modes in Pro Mode provide photographers with the ability to tailor their shots to specific lighting conditions and subjects, thereby enhancing the quality and composition of their photographs. For instance, the … | The different metering modes and focus modes in Pro Mode enhance photography by allowing photographers to tailor their settings to specific lighting conditions and subjects. Matrix metering is ideal for natural landscapes where light is distributed e… |

##### q\_id=456 · Causal · table

**Question:** How did the net carrying amounts of finite-lived intangible assets change from 2019 to 2020, and what could be the contributing factors?

**gold\_set:** ['image2', 'image6']

**Δ F1:** 0.226 → 0.485 (+0.259)

**Items dropped from top-10:**

- `text10` [TEXT, "Depreciation expense during the years ended December 31, 2020, 2019 and 2018 was…"]
- `text5` [TEXT, "Construction in progress is primarily comprised of construction of Gigafactory B…"]

**Items added to top-10:**

- `image2_text` [TWIN, src=image2]
- `image6_text` [TWIN, src=image6]

| | original | gold_redundant |
|---|---|---|
| prediction | The information provided does not explicitly mention the net carrying amounts of finite-lived intangible assets or how they changed from 2019 to 2020. Therefore, it's not possible to directly answer the question based solely on the given text. Howeve… | The net carrying amounts of finite-lived intangible assets decreased from 2019 to 2020. Specifically, the total net carrying amount of finite-lived intangible assets was $334 million in 2019 and $298 million in 2020, indicating a decrease of $36 mill… |

##### q\_id=30 · Comparative · table|text

**Question:** According to the report, which one is greater in population in the survey? Foreign born Latinos, or the Latinos interviewed by cellphone?

**gold\_set:** ['image1', 'image4', 'text4']

**Δ F1:** 0.167 → 0.421 (+0.254)

**Items dropped from top-10:**

- `text11` [TEXT, "For the full sample, a total of 705 respondents were U.S. born (including Puerto…"]
- `text12` [TEXT, "Samples for the low-incidence landline and low-incidence cell strata were drawn …"]

**Items added to top-10:**

- `image1_text` [TWIN, src=image1]
- `image4_text` [TWIN, src=image4]

| | original | gold_redundant |
|---|---|---|
| prediction | According to the report, the Latinos interviewed by cellphone are greater in population compared to the foreign-born Latinos. The report mentions that the sample consisted of a landline sampling frame (yielding 449 completed interviews) and a cellpho… | According to the report, the Latinos interviewed by cellphone are greater in population compared to the foreign-born Latinos. |

##### q\_id=0 · Descriptive · chart|text

**Question:** In 2015, how many percentage of surveyed adults, Republicans, and Democrats believed that the goverment was doing very/fairly well in reducing the threat of terrorism? Please write the answer in list format, e.g., ["3","2"]

**gold\_set:** ['image4', 'image7', 'text6', 'text9']

**Δ F1:** 0.000 → 0.400 (+0.400)

**Items dropped from top-10:**

- `text11` [TEXT, "Ideological divides are even starker, and  growing: About three-quarters   $(77\…"]

**Items added to top-10:**

- `image4_text` [TWIN, src=image4]

| | original | gold_redundant |
|---|---|---|
| prediction | ["63", "85"] | ["46", "34"] |

### Hurting cases  (original → negative\_redundant, F1 Δ < −0.1)

#### Aggregate statistics

| Metric | Value |
|---|---|
| n\_cases | 223 |
| avg twins added to top-10 | 2.206 |
| avg gold items displaced | 0.188 |
| avg noise (non-gold) displaced | 2.404 |
| avg Δ F1 | -0.2322 |

#### Representative examples (5)

##### q\_id=482 · Analytical · table|text

**Question:** How did the financial performance of the Sandoz segment change between 2020 and 2021 in terms of operating income and core operating income, and what were the main factors contributing to these changes?

**gold\_set:** ['image2', 'image3', 'text11', 'text6']

**Δ F1:** 0.422 → 0.306 (-0.116)

**Items dropped from top-10:**

- `text1` [TEXT, "Performance against the incentive targets, combined  with base salary and other …"]
- `text10` [TEXT, "The following table provides an overview of operating income from continuing ope…"]
- `text3` [TEXT, "Core operating income from continuing operations  was USD 16.6 billion   $(+8\%$…"]
- `text9` [TEXT, "Operating income was USD 10.7 billion   $(+17\%$  ,  $+15\%$   cc),  mainly driv…"]

**Items added to top-10:**

- `image1_text` [TWIN, src=image1]
- `image4_text` [TWIN, src=image4]
- `image6_text` [TWIN, src=image6]
- `image7_text` [TWIN, src=image7]

| | original | negative_redundant |
|---|---|---|
| prediction | The financial performance of the Sandoz segment improved between 2020 and 2021 in terms of both operating income and core operating income.  - **Operating Income:** The operating income for the Sandoz segment increased by 53% (+48% in constant curren… | The financial performance of the Sandoz segment showed the following changes between 2020 and 2021:  1. **Operating Income**:    - **2020**: 10,152 million USD    - **2021**: 11,689 million USD    - **Change in USD**: 15% increase    - **Change in co… |

##### q\_id=277 · Causal · chart|text

**Question:** How did the subscriber base and ARPU of Telkomsel change from 2013 to 2014, and what might have contributed to these changes?

**gold\_set:** ['image1', 'image5', 'image7', 'text7']

**Δ F1:** 0.505 → 0.324 (-0.181)

**Items dropped from top-10:**

- `image3` [IMAGE, images/indonesiamobilemarketresearch-ag-150106055934-conversion-gate02_95_image5.jpg]
- `text9` [TEXT, "Recently people use data-based IM,VolP,etc.thus leads to even less usage of sMs …"]

**Items added to top-10:**

- `image2_text` [TWIN, src=image2]
- `image6_text` [TWIN, src=image6]

| | original | negative_redundant |
|---|---|---|
| prediction | The subscriber base of Telkomsel increased from 132.7 million in 2013 to 139.3 million in 2014, while the ARPU decreased from 34.6 to 35.4. The increase in the subscriber base could be attributed to the growing demand for mobile services, while the d… | From 2013 to 2014, the subscriber base of Telkomsel decreased by 1.9 million, while the ARPU remained relatively stable. This could be attributed to a shift in user behavior towards more data-heavy services, which may have led to a slight dip in voic… |

##### q\_id=40 · Comparative · figure|text

**Question:** How did voter opinions on appointing opposition party members differ between Trump in 2016 and Obama in 2008?

**gold\_set:** ['image7', 'image8', 'text10', 'text7']

**Δ F1:** 0.462 → 0.257 (-0.204)

**Items dropped from top-10:**

- `image2` [IMAGE, images/11-21-16-Updated-Post-Election-Release_image10.jpg]
- `image4` [IMAGE, images/11-21-16-Updated-Post-Election-Release_image7.jpg]
- `image7` [GOLD, IMAGE, images/11-21-16-Updated-Post-Election-Release_image5.jpg]

**Items added to top-10:**

- `text3` [TEXT, "For their part, more than half of Republican and Republican-leaning voters   $(5…"]
- `text4` [TEXT, "However, few Trump voters have a positive  view of Trump reaching across partisa…"]
- `text9` [TEXT, "Half of voters say they are happy Donald  Trump was elected president, while abo…"]

| | original | negative_redundant |
|---|---|---|
| prediction | In 2008, Obama voters were 52% of voters saying Donald Trump should appoint Democrats to serve in important positions in his administration. | In 2016, only about a quarter (26%) of Trump voters said he should appoint Democrats to his cabinet, compared to 84% of Clinton voters who thought Trump's election would lead to worse race relations. This indicates a significant difference in opinion… |

##### q\_id=25 · Descriptive · figure

**Question:** List all countries, except U.S. and German, that are mentioned in some illustrated surveys of this report. Note that we do not count EU as a single country.

**gold\_set:** ['image2', 'image5', 'image6']

**Δ F1:** 1.000 → 0.471 (-0.529)

**Items dropped from top-10:**

- `text11` [TEXT, "For Germans of differing political stripes, the differences are less dramatic. S…"]
- `text4` [TEXT, "In the U.S., political affiliation dictates who people think is the most importa…"]

**Items added to top-10:**

- `image7_text` [TWIN, src=image7]
- `image8_text` [TWIN, src=image8]

| | original | negative_redundant |
|---|---|---|
| prediction | France, Japan, China, Russia, UK, Canada, Israel, Mexico, Austria | The countries mentioned in the surveys, excluding the U.S. and Germany, are:  1. France 2. Japan 3. China 4. Russia 5. UK 6. Canada 7. Israel 8. Mexico |

##### q\_id=1515 · Inferential · figure|text

**Question:** What is the angle in the right triangle shown in the diagram?

**gold\_set:** ['image4', 'text11']

**Δ F1:** 0.571 → 0.429 (-0.143)

**Items dropped from top-10:**

- `image7` [IMAGE, images/2309.17421v2_image18.jpg]
- `image8` [IMAGE, images/2309.17421v2_image16.jpg]

**Items added to top-10:**

- `image5_text` [TWIN, src=image5]
- `image7_text` [TWIN, src=image7]

| | original | negative_redundant |
|---|---|---|
| prediction | The angle in the right triangle shown in the diagram is approximately 26.57°. | The angle in the right triangle shown in the diagram is approximately \(26.57^\circ\). |

---

## Pipeline B

### Helping cases  (original → gold\_redundant, F1 Δ > +0.1)

#### Aggregate statistics

| Metric | Value |
|---|---|
| n\_cases | 359 |
| avg twins added to top-10 | 1.226 |
| avg gold items displaced | 0.058 |
| avg noise (non-gold) displaced | 1.162 |
| avg Δ F1 | +0.2605 |

#### Representative examples (5)

##### q\_id=324 · Analytical · figure|table|text

**Question:** Which method shows the best performance for news recommendation based on AUC and nDCG@10 metrics?

**gold\_set:** ['image4', 'image8', 'text1', 'text2', 'text6']

**Δ F1:** 0.333 → 0.471 (+0.137)

**Items dropped from top-10:**

- `text1` [GOLD, TEXT, "From the results we ﬁnd both LTUR and STUR are useful for news recommendation, a…"]

**Items added to top-10:**

- `image4_text` [TWIN, src=image4]

| | original | gold_redundant |
|---|---|---|
| prediction | The LSTUR-con method shows the best performance for news recommendation based on AUC and nDCG@10 metrics. | LSTUR-ini shows the best performance for news recommendation based on AUC and nDCG@10 metrics. |

##### q\_id=541 · Causal · table|text

**Question:** How did Shell Midstream Partners, L.P.'s operating income and cash from investing activities change from 2019 to 2020, and what might be the reasons for these changes?

**gold\_set:** ['image2', 'image6', 'text11', 'text2', 'text3']

**Δ F1:** 0.255 → 0.390 (+0.135)

**Items dropped from top-10:**

- `image1` [IMAGE, images/NYSE_SHLX_2020_image22.jpg]
- `text8` [TEXT, "In connection with the April 2020 Transaction, we also recorded contract assets …"]

**Items added to top-10:**

- `image2_text` [TWIN, src=image2]
- `image6_text` [TWIN, src=image6]

| | original | gold_redundant |
|---|---|---|
| prediction | Operating income decreased by $10 million in 2020 compared to the previous year, while cash from investing activities increased. The decrease was primarily due to lower transportation services revenue and higher operating costs related to planned tur… | Operating income decreased from $597 million in 2019 to $643.8 million, and cash provided by investing activities increased significantly due mainly the decrease of capital expenditures compared with previous years. |

##### q\_id=69 · Comparative · chart|table|text

**Question:** How do the priorities for Trump's presidency differ between Trump and Clinton voters, and what does this suggest about their views on his leadership?

**gold\_set:** ['image2', 'image5', 'text10', 'text6', 'text9']

**Δ F1:** 0.282 → 0.416 (+0.134)

**Items dropped from top-10:**

- `text4` [TEXT, "Another   $6\%$   of voters think Trump’s first  priority should be to change hi…"]

**Items added to top-10:**

- `image5_text` [TWIN, src=image5]

| | original | gold_redundant |
|---|---|---|
| prediction | Trump voters are more likely to name health care as Trump’s first priority (29% vs. 14%), while Clinton supporters prioritize the economy and terrorism over other issues, suggesting that they view his leadership through a different lens than do most … | Trump voters are more likely to name health care as Trump’s first priority, while Clinton supporters focus on the economy. This suggests that there is a significant difference in priorities between these two groups of people and indicates differing v… |

##### q\_id=25 · Descriptive · figure

**Question:** List all countries, except U.S. and German, that are mentioned in some illustrated surveys of this report. Note that we do not count EU as a single country.

**gold\_set:** ['image2', 'image5', 'image6']

**Δ F1:** 0.364 → 0.500 (+0.136)

**Items dropped from top-10:**

- `text10` [TEXT, "There are stark differences between and within the U.S. and Germany when it come…"]

**Items added to top-10:**

- `image5_text` [TWIN, src=image5]

| | original | gold_redundant |
|---|---|---|
| prediction | UK, China | UK, France, China |

##### q\_id=1145 · Inferential · chart

**Question:** Which country shows the highest percentage of uncertainty ('Don't know') about entrepreneurship among the options given?

**gold\_set:** ['image8']

**Δ F1:** 0.000 → 0.296 (+0.296)

**Items dropped from top-10:**

- `text8` [TEXT, "ENTREPRENEURSHIP BY COUNTRY"]

**Items added to top-10:**

- `image8_text` [TWIN, src=image8]

| | original | gold_redundant |
|---|---|---|
| prediction | Libya | Palestine shows the highest percentage of uncertainty ('Don't know') about entrepreneurship among all countries, with 21% indicating they don’t have a clear answer. |

### Hurting cases  (original → negative\_redundant, F1 Δ < −0.1)

#### Aggregate statistics

| Metric | Value |
|---|---|
| n\_cases | 243 |
| avg twins added to top-10 | 2.848 |
| avg gold items displaced | 0.210 |
| avg noise (non-gold) displaced | 2.609 |
| avg Δ F1 | -0.2335 |

#### Representative examples (5)

##### q\_id=401 · Analytical · table|text

**Question:** How do different data augmentation techniques impact the performance of BERT models on the QOP dataset, and how is this effect measured across various sentiment analysis and named entity recognition tasks?

**gold\_set:** ['image2', 'image4', 'text10', 'text2', 'text9']

**Δ F1:** 0.523 → 0.329 (-0.194)

**Items dropped from top-10:**

- `text1` [TEXT, "Results are shown in Table  8 . We ﬁrst look at the ﬁrst line, with all results …"]
- `text10` [GOLD, TEXT, "As mentioned in Section  3.3 , Tversky index (TI) offers the ﬂexibility in contr…"]
- `text5` [TEXT, "the data augmented technique are exactly the same as  +negative , except that we…"]
- `text6` [TEXT, "Sudre et al.  ( 2017 ) addressed the severe class imbal- ance issue for the imag…"]
- `text7` [TEXT, "Data imbalance results in the following two issues: (1)  the training-test discr…"]
- `text9` [GOLD, TEXT, "DSC achieves the highest F1 score across all datasets. Specially, for  $^+$  pos…"]

**Items added to top-10:**

- `image1_text` [TWIN, src=image1]
- `image3_text` [TWIN, src=image3]
- `image5_text` [TWIN, src=image5]
- `image6_text` [TWIN, src=image6]
- `image7_text` [TWIN, src=image7]
- `image8_text` [TWIN, src=image8]

| | original | negative_redundant |
|---|---|---|
| prediction | Different data augmentation techniques impact the performance of BERT models on QQP dataset, and this effect is measured across various sentiment analysis (SST-2) named entity recognition tasks. | Different data augmentation techniques impact the performance of BERT models on various sentiment analysis and named entity recognition tasks. The effect is measured by comparing F1 scores for different variants, such as +FL (small improvement), DL (… |

##### q\_id=1591 · Application-based · figure|text

**Question:** Which areas do candidates need to focus on for the LinkedIn Recruiter Certification exam, and how does understanding Boolean search results through a Venn diagram relate to these topic areas?

**gold\_set:** ['image7', 'text4']

**Δ F1:** 0.513 → 0.378 (-0.134)

**Items dropped from top-10:**

- `text2` [TEXT, "Engineering AND Java NOT Manager"]

**Items added to top-10:**

- `image4_text` [TWIN, src=image4]

| | original | negative_redundant |
|---|---|---|
| prediction | Candidates need to focus on the entire recruiting life cycle, including building a talent pipeline and posting jobs. Understanding Boolean search results through Venn diagrams relates as it is an important skill for all recruiters in efficiently sear… | Candidates need to focus on the following areas for LinkedIn Recruiter Certification: identifying talent through search, engaging with potential candidates using their presence and InMail features of LinkedIn; building a pipeline by creating an effec… |

##### q\_id=544 · Causal · table|text

**Question:** How did the free cash flow change from 2020 to 2021, and what were the main contributing factors according to the financial statements?

**gold\_set:** ['image5', 'image8', 'text3', 'text5', 'text6', 'text9']

**Δ F1:** 0.518 → 0.342 (-0.175)

**Items dropped from top-10:**

- `text2` [TEXT, "These decreases were partly offset by an increase  in financial and other non-cu…"]
- `text4` [TEXT, "Total current assets of USD 45.7 billion at December  31, 2021, increased by USD…"]

**Items added to top-10:**

- `image2_text` [TWIN, src=image2]
- `image6_text` [TWIN, src=image6]

| | original | negative_redundant |
|---|---|---|
| prediction | Free cash flow increased by USD 1.6 billion from 2020 to 2021, mainly driven by higher operating income adjusted for non-cash items and lower payments out of provisions due to legal matters in the prior year; however it was partly offset by a upfront… | Free cash flow increased to USD 13.3 billion in the year ended December 20, compared with $USD (+)$+9\%) of free cash flows for fiscal years ending on Decembe r-December -86; and (Decemeber-Decemebr ) respectively. The increase was mainly driven by h… |

##### q\_id=46 · Comparative · figure|text

**Question:** How did the percentages for Republicans and Democrats wanting their leaders to 'stand up' to opposition change from January 2018 to January 2019?

**gold\_set:** ['image7', 'image8', 'text1']

**Δ F1:** 0.279 → 0.138 (-0.141)

**Items dropped from top-10:**

- `text6` [TEXT, "In both parties, increasing shares want  leaders to ‘stand up’ to opposition"]
- `text9` [TEXT, "Today, as many Republicans  and Republican leaners say  they are dissatisfied wi…"]

**Items added to top-10:**

- `image1_text` [TWIN, src=image1]
- `image5_text` [TWIN, src=image5]

| | original | negative_redundant |
|---|---|---|
| prediction | The percentage of Republicans wanting their leaders to 'stand up' increased from 40% in January 2018, while the Democrats wanted this by a margin that grew slightly more than two-to-one. | The percentage of Republicans wanting their leaders to "stand up" more against opposition increased from 38% in January 2017, dropped slightly but then rose again by the end of that year. In contrast, Democrats' desire for a stronger stance on opposi… |

##### q\_id=303 · Descriptive · figure

**Question:** Which figures include line plots in the paper?

**gold\_set:** ['image2', 'image3']

**Δ F1:** 0.333 → 0.143 (-0.190)

**Items dropped from top-10:**

- `text5` [TEXT, "Recent multimodal foundation models are very widely adopted but still model diff…"]
- `text7` [TEXT, "We found that the standard LLaMa architecture showed complex divergences due to …"]

**Items added to top-10:**

- `image4_text` [TWIN, src=image4]
- `image5_text` [TWIN, src=image5]

| | original | negative_redundant |
|---|---|---|
| prediction | Figures 5a, Figure 6c and figure of the paper. | Figures 6a and Figure 5b include line plots in the paper. |

---
