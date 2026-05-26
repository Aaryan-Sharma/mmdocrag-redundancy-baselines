# How Redundancy Affects Answers

> **TL;DR.** Across all variant comparisons, 9,368 question-level predictions changed from the original. Of those, 1991 improved by >0.1 F1 and 1243 degraded. In 3767 cases the variant's retrieved top-10 contained both an image quote and its text twin simultaneously (potential double-citation). Gold-redundant variants produce the most improvements; negative-redundant produces the most degradations.

---

## Pipeline A

### Change summary

| Variant pair | n\_changed | n\_improved (ΔF1>0.1) | n\_degraded (ΔF1<−0.1) | n\_equiv | n\_double\_cite |
|---|---|---|---|---|---|
| original → gold_redundant | 1640 | 464 | 133 | 1043 | 630 |
| original → negative_redundant | 1644 | 166 | 223 | 1255 | 520 |
| original → mixed_redundant | 1829 | 454 | 187 | 1188 | 889 |

### original → gold_redundant

#### Improved cases (464 total, 5 shown)

##### q\_id=323 · Analytical · table|text

**Question:** Which model achieves the highest F1-value for Named Entity Recognition (NER) and what features contribute to it?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image3', 'image4', 'text9'] | ['image3', 'image3_text', 'image4', 'image4_text', 'text9'] |
| retrieved | `image1`, `text5`, `text11`, `text8`, `text4`, `text12`, `image5`, `text1`, `text7`, `text10` | `image1`, `image3_text` **[GOLD,TWIN]**, `text5`, `text11`, `text8`, `image5`, `text4`, `text12`, `text1`, `image4_text` **[GOLD,TWIN]** |
| prediction | The model achieving the highest F1-value for Named Entity Recognition (NER) is the one built on the $\mathrm{NCRF++}$ toolkit, which utilizes both character sequence and word sequence representations. This model incorporates both handcrafted features… | The model that achieves the highest F1-value for Named Entity Recognition (NER) is CCNN+WLSTM+CRF, with an F1-value of 91.35. This model uses the combination of CCNN, WLSTM, and CRF layers, which contributes to its superior performance in NER tasks. |
| F1 | 0.215 | 0.519 (Δ+0.303) |

##### q\_id=1995 · Application-based · table

**Question:** How do the different metering modes and focus modes in Pro Mode enhance photography under various scenarios?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image4', 'image5'] | ['image4', 'image4_text', 'image5', 'image5_text'] |
| retrieved | `image3`, `text9`, `text10`, `text12`, `text3`, `text1`, `text6`, `image2`, `text7`, `text8` | `image3`, `image5_text` **[GOLD,TWIN]**, `text9`, `text10`, `text12`, `image4_text` **[GOLD,TWIN]**, `text3`, `text1`, `text6`, `image2` |
| prediction | The different metering modes and focus modes in Pro Mode provide photographers with the ability to tailor their shots to specific lighting conditions and subjects, thereby enhancing the quality and composition of their photographs. For instance, the … | The different metering modes and focus modes in Pro Mode enhance photography by allowing photographers to tailor their settings to specific lighting conditions and subjects. Matrix metering is ideal for natural landscapes where light is distributed e… |
| F1 | 0.173 | 0.358 (Δ+0.184) |

##### q\_id=456 · Causal · table

**Question:** How did the net carrying amounts of finite-lived intangible assets change from 2019 to 2020, and what could be the contributing factors?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image2', 'image6'] | ['image2', 'image2_text', 'image6', 'image6_text'] |
| retrieved | `text8`, `text4`, `text3`, `text9`, `text2`, `text1`, `text6`, `text7`, `text5`, `text10` | `image2_text` **[GOLD,TWIN]**, `text8`, `text4`, `text3`, `text9`, `text2`, `text1`, `text6`, `image6_text` **[GOLD,TWIN]**, `text7` |
| prediction | The information provided does not explicitly mention the net carrying amounts of finite-lived intangible assets or how they changed from 2019 to 2020. Therefore, it's not possible to directly answer the question based solely on the given text. Howeve… | The net carrying amounts of finite-lived intangible assets decreased from 2019 to 2020. Specifically, the total net carrying amount of finite-lived intangible assets was $334 million in 2019 and $298 million in 2020, indicating a decrease of $36 mill… |
| F1 | 0.226 | 0.485 (Δ+0.259) |

##### q\_id=30 · Comparative · table|text

**Question:** According to the report, which one is greater in population in the survey? Foreign born Latinos, or the Latinos interviewed by cellphone?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image1', 'image4', 'text4'] | ['image1', 'image1_text', 'image4', 'image4_text', 'text4'] |
| retrieved | `image6`, `text5`, `text10`, `image7`, `text2`, `image2`, `text4` **[GOLD]**, `image5`, `text12`, `text11` | `image6`, `text5`, `text10`, `image4_text` **[GOLD,TWIN]**, `image7`, `text2`, `image2`, `text4` **[GOLD]**, `image1_text` **[GOLD,TWIN]**, `image5` |
| prediction | According to the report, the Latinos interviewed by cellphone are greater in population compared to the foreign-born Latinos. The report mentions that the sample consisted of a landline sampling frame (yielding 449 completed interviews) and a cellpho… | According to the report, the Latinos interviewed by cellphone are greater in population compared to the foreign-born Latinos. |
| F1 | 0.167 | 0.421 (Δ+0.254) |

##### q\_id=0 · Descriptive · chart|text

**Question:** In 2015, how many percentage of surveyed adults, Republicans, and Democrats believed that the goverment was doing very/fairly well in reducing the threat of terrorism? Please write the answer in list format, e.g., ["3","2"]

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image4', 'image7', 'text6', 'text9'] | ['image4', 'image4_text', 'image7', 'image7_text', 'text6', 'text9'] |
| retrieved | `image5`, `text9` **[GOLD]**, `text1`, `text6` **[GOLD]**, `text4`, `text5`, `text2`, `text12`, `text3`, `text11` | `image5`, `text9` **[GOLD]**, `text1`, `text6` **[GOLD]**, `text4`, `text5`, `text2`, `image4_text` **[GOLD,TWIN]**, `text12`, `text3` |
| prediction | ["63", "85"] | ["46", "34"] |
| F1 | 0.000 | 0.400 (Δ+0.400) |

#### Degraded cases (133 total, 5 shown)

##### q\_id=385 · Analytical · figure|table|text · **DOUBLE-CITE**

**Question:** How do different methods, including Step-Back and RAG, influence performance in various QA tasks compared to GPT-4, and what are the common error types associated with Step-Back Prompting?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image1', 'image4', 'image5', 'image8', 'text1', 'text10'] | ['image1', 'image1_text', 'image4', 'image4_text', 'image5', 'image5_text', 'image8', 'image8_text', 'text1', 'text10'] |
| retrieved | `image3`, `text9`, `text5`, `image1` **[GOLD]**, `text3`, `image6`, `image8` **[GOLD]**, `text12`, `text8`, `text11` | `image3`, `text9`, `text5`, `image1` **[GOLD]**, `image8_text` **[GOLD,TWIN]**, `text3`, `image6`, `image8` **[GOLD]**, `text12`, `text8` |
| prediction | Step-Back Prompting and RAG together show strong performance gains across various Knowledge QA tasks, outperforming GPT-4 in some cases but falling short in others. The common error types associated with Step-Back Prompting include reasoning errors a… | Step-Back Prompting and RAG generally show strong performance across various Knowledge QA tasks, often outperforming GPT-4. However, their effectiveness varies depending on the specific task. For instance, on TimeQA and StrategyQA, Step-Back Promptin… |
| F1 | 0.514 | 0.380 (Δ-0.134) |

##### q\_id=277 · Causal · chart|text · **DOUBLE-CITE**

**Question:** How did the subscriber base and ARPU of Telkomsel change from 2013 to 2014, and what might have contributed to these changes?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image1', 'image5', 'image7', 'text7'] | ['image1', 'image1_text', 'image5', 'image5_text', 'image7', 'image7_text', 'text7'] |
| retrieved | `image2`, `text11`, `image1` **[GOLD]**, `image7` **[GOLD]**, `image6`, `image3`, `text8`, `text5`, `text1`, `text9` | `image2`, `image1_text` **[GOLD,TWIN]**, `text11`, `image1` **[GOLD]**, `image7` **[GOLD]**, `image7_text` **[GOLD,TWIN]**, `image6`, `image3`, `image5_text` **[GOLD,TWIN]**, `text8` |
| prediction | The subscriber base of Telkomsel increased from 132.7 million in 2013 to 139.3 million in 2014, while the ARPU decreased from 34.6 to 35.4. The increase in the subscriber base could be attributed to the growing demand for mobile services, while the d… | From 2013 to 2014, the subscriber base of Telkomsel increased by 13.3 million, reaching 139.3 million. The ARPU of Telkomsel also saw an increase, rising from 34.6 to 35.4 thousand. These changes could be attributed to several factors such as success… |
| F1 | 0.505 | 0.296 (Δ-0.210) |

##### q\_id=39 · Comparative · chart|text · **DOUBLE-CITE**

**Question:** How have the political orientations of Republican and Democratic voters shifted from 2008 to 2016?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image4', 'image8', 'text11', 'text4', 'text9'] | ['image4', 'image4_text', 'image8', 'image8_text', 'text11', 'text4', 'text9'] |
| retrieved | `image2`, `text11` **[GOLD]**, `text3`, `image4` **[GOLD]**, `image8` **[GOLD]**, `text9` **[GOLD]**, `text1`, `image7`, `text5`, `text6` | `image2`, `text11` **[GOLD]**, `image4_text` **[GOLD,TWIN]**, `image8_text` **[GOLD,TWIN]**, `text3`, `image4` **[GOLD]**, `image8` **[GOLD]**, `text9` **[GOLD]**, `text1`, `image7` |
| prediction | The political orientations of Republican and Democratic voters have shifted from 2008 to 2016. The Republican voters have become more conservative, while the Democratic voters have become more liberal. | From 2008 to 2016, there has been a shift in the political orientations of both Republican and Democratic voters. While Republican voters have consistently favored a more conservative direction, the percentage of Democratic voters who support a more … |
| F1 | 0.381 | 0.154 (Δ-0.227) |

##### q\_id=25 · Descriptive · figure · **DOUBLE-CITE**

**Question:** List all countries, except U.S. and German, that are mentioned in some illustrated surveys of this report. Note that we do not count EU as a single country.

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image2', 'image5', 'image6'] | ['image2', 'image2_text', 'image5', 'image5_text', 'image6', 'image6_text'] |
| retrieved | `image7`, `text12`, `text8`, `image5` **[GOLD]**, `image6` **[GOLD]**, `image4`, `text3`, `text11`, `text4`, `image8` | `image7`, `image2_text` **[GOLD,TWIN]**, `image5` **[GOLD]**, `image6` **[GOLD]**, `image4`, `image5_text` **[GOLD,TWIN]**, `image6_text` **[GOLD,TWIN]**, `text12`, `text8`, `image8` |
| prediction | France, Japan, China, Russia, UK, Canada, Israel, Mexico, Austria | Based on the provided information, the countries mentioned in the surveys, excluding the U.S. and Germany, are:  1. France 2. Japan 3. Russia 4. China 5. UK 6. Canada 7. Israel 8. Mexico |
| F1 | 1.000 | 0.421 (Δ-0.579) |

##### q\_id=174 · Inferential · figure|text · **DOUBLE-CITE**

**Question:** How do perceptions of social media content and behavior differ between men and women, and what are the implications for how social media platforms can tailor their recommendations or advertisements based on these differences?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image2', 'image4', 'text3', 'text6'] | ['image2', 'image2_text', 'image4', 'image4_text', 'text3', 'text6'] |
| retrieved | `image4` **[GOLD]**, `text3` **[GOLD]**, `image8`, `text6` **[GOLD]**, `image2` **[GOLD]**, `image3`, `text1`, `text7`, `text5`, `text8` | `image4` **[GOLD]**, `text3` **[GOLD]**, `image8`, `image4_text` **[GOLD,TWIN]**, `text6` **[GOLD]**, `image2` **[GOLD]**, `image3`, `text1`, `text7`, `text5` |
| prediction | The provided data indicates that men are slightly more likely than women to encounter harassing or abusive behavior online, particularly in terms of seeing people being mean or bullying on social media. Conversely, women are more likely to see kind o… | Perceptions of social media content and behavior vary between men and women, with men more likely to encounter bullying and deception compared to women. Women, on the other hand, tend to see more kindness and support. These differences suggest that s… |
| F1 | 0.522 | 0.410 (Δ-0.112) |

### original → negative_redundant

#### Improved cases (166 total, 5 shown)

##### q\_id=340 · Analytical · chart|table|text

**Question:** What are the primary reasons for refuted and NEI claims in ScITaB, and how do these reasons impact the performance of different models in zero-shot 3-class classification?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image1', 'image6', 'image7', 'text3', 'text5', 'text7'] | ['image1', 'image6', 'image7', 'text3', 'text5', 'text7'] |
| retrieved | `image7` **[GOLD]**, `text10`, `text7` **[GOLD]**, `text12`, `image4`, `text5` **[GOLD]**, `text3` **[GOLD]**, `text6`, `text8`, `text1` | `image7` **[GOLD]**, `text10`, `text7` **[GOLD]**, `text12`, `text5` **[GOLD]**, `text3` **[GOLD]**, `text6`, `text8`, `text1`, `image4` |
| prediction | The primary reasons for refuted claims in ScITaB include insufficient evidence in the table, lack of background knowledge, and the use of vague pronouns that bring ambiguity to the claim. For NEI claims, the main reasons are insufficient evidence in … | The primary reasons for refuted and NEI claims in ScITaB include insufficient evidence in the table, the lack of background knowledge, the lack of closed-domain knowledge necessitating additional information in the paper, and the use of vague pronoun… |
| F1 | 0.358 | 0.462 (Δ+0.103) |

##### q\_id=1995 · Application-based · table · **DOUBLE-CITE**

**Question:** How do the different metering modes and focus modes in Pro Mode enhance photography under various scenarios?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image4', 'image5'] | ['image4', 'image5'] |
| retrieved | `image3`, `text9`, `text10`, `text12`, `text3`, `text1`, `text6`, `image2`, `text7`, `text8` | `image3`, `text9`, `image3_text` **[TWIN]**, `text10`, `text12`, `text3`, `text1`, `text6`, `text7`, `text8` |
| prediction | The different metering modes and focus modes in Pro Mode provide photographers with the ability to tailor their shots to specific lighting conditions and subjects, thereby enhancing the quality and composition of their photographs. For instance, the … | The different metering modes and focus modes in Pro Mode enhance photography by allowing photographers to tailor their settings to specific lighting conditions and subjects. Metering modes like spot, center-weighted, and matrix help the camera measur… |
| F1 | 0.173 | 0.283 (Δ+0.110) |

##### q\_id=240 · Causal · figure|table|text

**Question:** How does the increase in weekday ridership growth in Mountain View and Palo Alto relate to the current capacity issues on trains?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image2', 'image6', 'image7', 'text11'] | ['image2', 'image6', 'image7', 'text11'] |
| retrieved | `image4`, `text11` **[GOLD]**, `text8`, `image3`, `text2`, `image2` **[GOLD]**, `image1`, `text9`, `image7` **[GOLD]**, `text4` | `image4`, `text11` **[GOLD]**, `text8`, `image3`, `text2`, `image2` **[GOLD]**, `text9`, `text4`, `image1`, `image7` **[GOLD]** |
| prediction | The increase in weekday ridership growth in Mountain View and Palo Alto is related to the current capacity issues on trains because of the rapid growth in these areas, which has led to an increase in demand for public transportation services like Cal… | The increase in weekday ridership growth in Mountain View and Palo Alto is related to the current capacity issues on trains because of the rapid growth in these areas, which has led to an increase in demand for transportation services, including trai… |
| F1 | 0.336 | 0.460 (Δ+0.124) |

##### q\_id=182 · Comparative · figure|text · **DOUBLE-CITE**

**Question:** How do perceptions of economic conditions and job availability differ between political affiliations, and what trends are observed over time?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image2', 'image5', 'image7', 'text1', 'text3'] | ['image2', 'image5', 'image7', 'text1', 'text3'] |
| retrieved | `image4`, `text9`, `text11`, `text6`, `text3` **[GOLD]**, `text10`, `image7` **[GOLD]**, `image5` **[GOLD]**, `image2` **[GOLD]**, `text2` | `image4`, `text9`, `text11`, `text6`, `image4_text` **[TWIN]**, `text3` **[GOLD]**, `text10`, `image7` **[GOLD]**, `image5` **[GOLD]**, `image2` **[GOLD]** |
| prediction | There is a significant partisan gap in views of job availability, with Republicans being more likely than Democrats to believe that there are plenty of jobs available (71% vs. 53%). This trend has been consistent over time, with Republicans consisten… | Perceptions of economic conditions and job availability show significant differences between political affiliations, with Republicans generally more optimistic about job availability than Democrats. Over time, there has been an overall increase in th… |
| F1 | 0.209 | 0.414 (Δ+0.205) |

##### q\_id=0 · Descriptive · chart|text · **DOUBLE-CITE**

**Question:** In 2015, how many percentage of surveyed adults, Republicans, and Democrats believed that the goverment was doing very/fairly well in reducing the threat of terrorism? Please write the answer in list format, e.g., ["3","2"]

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image4', 'image7', 'text6', 'text9'] | ['image4', 'image7', 'text6', 'text9'] |
| retrieved | `image5`, `text9` **[GOLD]**, `text1`, `text6` **[GOLD]**, `text4`, `text5`, `text2`, `text12`, `text3`, `text11` | `image5`, `text9` **[GOLD]**, `text1`, `text6` **[GOLD]**, `text4`, `text5`, `text2`, `image5_text` **[TWIN]**, `image8_text` **[TWIN]**, `image3_text` **[TWIN]** |
| prediction | ["63", "85"] | ["46", "64"] |
| F1 | 0.000 | 0.800 (Δ+0.800) |

#### Degraded cases (223 total, 5 shown)

##### q\_id=482 · Analytical · table|text

**Question:** How did the financial performance of the Sandoz segment change between 2020 and 2021 in terms of operating income and core operating income, and what were the main factors contributing to these changes?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image2', 'image3', 'text11', 'text6'] | ['image2', 'image3', 'text11', 'text6'] |
| retrieved | `text4`, `text2`, `text12`, `text8`, `text6` **[GOLD]**, `text11` **[GOLD]**, `text3`, `text9`, `text10`, `text1` | `image6_text` **[TWIN]**, `text4`, `image4_text` **[TWIN]**, `text2`, `image7_text` **[TWIN]**, `image1_text` **[TWIN]**, `text12`, `text8`, `text6` **[GOLD]**, `text11` **[GOLD]** |
| prediction | The financial performance of the Sandoz segment improved between 2020 and 2021 in terms of both operating income and core operating income.  - **Operating Income:** The operating income for the Sandoz segment increased by 53% (+48% in constant curren… | The financial performance of the Sandoz segment showed the following changes between 2020 and 2021:  1. **Operating Income**:    - **2020**: 10,152 million USD    - **2021**: 11,689 million USD    - **Change in USD**: 15% increase    - **Change in co… |
| F1 | 0.422 | 0.306 (Δ-0.116) |

##### q\_id=277 · Causal · chart|text · **DOUBLE-CITE**

**Question:** How did the subscriber base and ARPU of Telkomsel change from 2013 to 2014, and what might have contributed to these changes?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image1', 'image5', 'image7', 'text7'] | ['image1', 'image5', 'image7', 'text7'] |
| retrieved | `image2`, `text11`, `image1` **[GOLD]**, `image7` **[GOLD]**, `image6`, `image3`, `text8`, `text5`, `text1`, `text9` | `image2`, `image6_text` **[TWIN]**, `text11`, `image2_text` **[TWIN]**, `image1` **[GOLD]**, `image7` **[GOLD]**, `text8`, `image6`, `text5`, `text1` |
| prediction | The subscriber base of Telkomsel increased from 132.7 million in 2013 to 139.3 million in 2014, while the ARPU decreased from 34.6 to 35.4. The increase in the subscriber base could be attributed to the growing demand for mobile services, while the d… | From 2013 to 2014, the subscriber base of Telkomsel decreased by 1.9 million, while the ARPU remained relatively stable. This could be attributed to a shift in user behavior towards more data-heavy services, which may have led to a slight dip in voic… |
| F1 | 0.505 | 0.324 (Δ-0.181) |

##### q\_id=40 · Comparative · figure|text

**Question:** How did voter opinions on appointing opposition party members differ between Trump in 2016 and Obama in 2008?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image7', 'image8', 'text10', 'text7'] | ['image7', 'image8', 'text10', 'text7'] |
| retrieved | `image8` **[GOLD]**, `text10` **[GOLD]**, `text7` **[GOLD]**, `image5`, `image7` **[GOLD]**, `image2`, `text12`, `text1`, `text11`, `image4` | `image8` **[GOLD]**, `text10` **[GOLD]**, `text7` **[GOLD]**, `text12`, `text1`, `text11`, `image5`, `text3`, `text4`, `text9` |
| prediction | In 2008, Obama voters were 52% of voters saying Donald Trump should appoint Democrats to serve in important positions in his administration. | In 2016, only about a quarter (26%) of Trump voters said he should appoint Democrats to his cabinet, compared to 84% of Clinton voters who thought Trump's election would lead to worse race relations. This indicates a significant difference in opinion… |
| F1 | 0.462 | 0.257 (Δ-0.204) |

##### q\_id=25 · Descriptive · figure · **DOUBLE-CITE**

**Question:** List all countries, except U.S. and German, that are mentioned in some illustrated surveys of this report. Note that we do not count EU as a single country.

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image2', 'image5', 'image6'] | ['image2', 'image5', 'image6'] |
| retrieved | `image7`, `text12`, `text8`, `image5` **[GOLD]**, `image6` **[GOLD]**, `image4`, `text3`, `text11`, `text4`, `image8` | `image7`, `image7_text` **[TWIN]**, `image5` **[GOLD]**, `image6` **[GOLD]**, `image4`, `text12`, `text8`, `image8_text` **[TWIN]**, `text3`, `image8` |
| prediction | France, Japan, China, Russia, UK, Canada, Israel, Mexico, Austria | The countries mentioned in the surveys, excluding the U.S. and Germany, are:  1. France 2. Japan 3. China 4. Russia 5. UK 6. Canada 7. Israel 8. Mexico |
| F1 | 1.000 | 0.471 (Δ-0.529) |

##### q\_id=1515 · Inferential · figure|text · **DOUBLE-CITE**

**Question:** What is the angle in the right triangle shown in the diagram?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image4', 'text11'] | ['image4', 'text11'] |
| retrieved | `image4` **[GOLD]**, `text3`, `text11` **[GOLD]**, `text9`, `text4`, `image5`, `text7`, `text2`, `image8`, `image7` | `image4` **[GOLD]**, `text3`, `text11` **[GOLD]**, `text9`, `text4`, `image5`, `text7`, `text2`, `image5_text` **[TWIN]**, `image7_text` **[TWIN]** |
| prediction | The angle in the right triangle shown in the diagram is approximately 26.57°. | The angle in the right triangle shown in the diagram is approximately \(26.57^\circ\). |
| F1 | 0.571 | 0.429 (Δ-0.143) |

### original → mixed_redundant

#### Improved cases (454 total, 5 shown)

##### q\_id=323 · Analytical · table|text

**Question:** Which model achieves the highest F1-value for Named Entity Recognition (NER) and what features contribute to it?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image3', 'image4', 'text9'] | ['image3', 'image3_text', 'image4', 'image4_text', 'text9'] |
| retrieved | `image1`, `text5`, `text11`, `text8`, `text4`, `text12`, `image5`, `text1`, `text7`, `text10` | `image1`, `image3_text` **[GOLD,TWIN]**, `text5`, `text11`, `text8`, `text4`, `text12`, `image5`, `text1`, `image4_text` **[GOLD,TWIN]** |
| prediction | The model achieving the highest F1-value for Named Entity Recognition (NER) is the one built on the $\mathrm{NCRF++}$ toolkit, which utilizes both character sequence and word sequence representations. This model incorporates both handcrafted features… | The model that achieves the highest F1-value for Named Entity Recognition (NER) is CCNN+WLSTM+CRF, with an F1-value of 91.35. This model uses the combination of CCNN, WLSTM, and CRF features. |
| F1 | 0.215 | 0.533 (Δ+0.318) |

##### q\_id=1995 · Application-based · table · **DOUBLE-CITE**

**Question:** How do the different metering modes and focus modes in Pro Mode enhance photography under various scenarios?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image4', 'image5'] | ['image4', 'image4_text', 'image5', 'image5_text'] |
| retrieved | `image3`, `text9`, `text10`, `text12`, `text3`, `text1`, `text6`, `image2`, `text7`, `text8` | `image3`, `image5_text` **[GOLD,TWIN]**, `text9`, `image3_text` **[TWIN]**, `text10`, `text12`, `image4_text` **[GOLD,TWIN]**, `text3`, `text1`, `text6` |
| prediction | The different metering modes and focus modes in Pro Mode provide photographers with the ability to tailor their shots to specific lighting conditions and subjects, thereby enhancing the quality and composition of their photographs. For instance, the … | The different metering modes and focus modes in Pro Mode enhance photography by allowing photographers to tailor their settings to specific lighting and subject conditions, thereby achieving optimal exposure and focus.   - **Metering Modes**:   - **M… |
| F1 | 0.173 | 0.276 (Δ+0.103) |

##### q\_id=240 · Causal · figure|table|text

**Question:** How does the increase in weekday ridership growth in Mountain View and Palo Alto relate to the current capacity issues on trains?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image2', 'image6', 'image7', 'text11'] | ['image2', 'image2_text', 'image6', 'image6_text', 'image7', 'image7_text', 'text11'] |
| retrieved | `image4`, `text11` **[GOLD]**, `text8`, `image3`, `text2`, `image2` **[GOLD]**, `image1`, `text9`, `image7` **[GOLD]**, `text4` | `image4`, `text11` **[GOLD]**, `text8`, `image3`, `text2`, `image2` **[GOLD]**, `text9`, `text4`, `image1`, `image7` **[GOLD]** |
| prediction | The increase in weekday ridership growth in Mountain View and Palo Alto is related to the current capacity issues on trains because of the rapid growth in these areas, which has led to an increase in demand for public transportation services like Cal… | The increase in weekday ridership growth in Mountain View and Palo Alto is related to the current capacity issues on trains because of the rapid growth in these areas, which has led to an increase in demand for transportation services, including trai… |
| F1 | 0.336 | 0.460 (Δ+0.124) |

##### q\_id=54 · Comparative · chart · **DOUBLE-CITE**

**Question:** Which category contributes the most to the number of transactions and which to the gross margin value in online retail?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image1', 'image3'] | ['image1', 'image1_text', 'image3', 'image3_text'] |
| retrieved | `image1` **[GOLD]**, `text11`, `text5`, `text7`, `image3` **[GOLD]**, `text2`, `image2`, `image4`, `text10`, `text3` | `image1` **[GOLD]**, `text11`, `image3_text` **[GOLD,TWIN]**, `image1_text` **[GOLD,TWIN]**, `text5`, `text7`, `image3` **[GOLD]**, `text2`, `image5_text` **[TWIN]**, `image2_text` **[TWIN]** |
| prediction | The category that contributes the most to the number of transactions in online retail is "Fashion, Footwear & Accessories," as it accounts for 35% of the total transactions according to the pie chart. For the gross margin value, the category contribu… | The category that contributes the most to the number of transactions is "Fashion, Footwear & Accessories" at 35%. The category that contributes the most to the gross margin value is "Mobile, Tablets & Accessories" at 35%. |
| F1 | 0.471 | 0.681 (Δ+0.210) |

##### q\_id=0 · Descriptive · chart|text · **DOUBLE-CITE**

**Question:** In 2015, how many percentage of surveyed adults, Republicans, and Democrats believed that the goverment was doing very/fairly well in reducing the threat of terrorism? Please write the answer in list format, e.g., ["3","2"]

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image4', 'image7', 'text6', 'text9'] | ['image4', 'image4_text', 'image7', 'image7_text', 'text6', 'text9'] |
| retrieved | `image5`, `text9` **[GOLD]**, `text1`, `text6` **[GOLD]**, `text4`, `text5`, `text2`, `text12`, `text3`, `text11` | `image5`, `text9` **[GOLD]**, `text1`, `text6` **[GOLD]**, `text4`, `text5`, `text2`, `image5_text` **[TWIN]**, `image8_text` **[TWIN]**, `image3_text` **[TWIN]** |
| prediction | ["63", "85"] | ["46", "64"] |
| F1 | 0.000 | 0.800 (Δ+0.800) |

#### Degraded cases (187 total, 5 shown)

##### q\_id=385 · Analytical · figure|table|text

**Question:** How do different methods, including Step-Back and RAG, influence performance in various QA tasks compared to GPT-4, and what are the common error types associated with Step-Back Prompting?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image1', 'image4', 'image5', 'image8', 'text1', 'text10'] | ['image1', 'image1_text', 'image4', 'image4_text', 'image5', 'image5_text', 'image8', 'image8_text', 'text1', 'text10'] |
| retrieved | `image3`, `text9`, `text5`, `image1` **[GOLD]**, `text3`, `image6`, `image8` **[GOLD]**, `text12`, `text8`, `text11` | `image3`, `text9`, `text5`, `image1` **[GOLD]**, `image8_text` **[GOLD,TWIN]**, `text3`, `text12`, `image6`, `text8`, `text11` |
| prediction | Step-Back Prompting and RAG together show strong performance gains across various Knowledge QA tasks, outperforming GPT-4 in some cases but falling short in others. The common error types associated with Step-Back Prompting include reasoning errors a… | Different methods, including Step-Back Prompting and Retrieval-Augmented Generation (RAG), show varying degrees of performance improvement over GPT-4 across different QA tasks. For instance, on the MMLU Physics and Chemistry benchmarks, Step-Back Pro… |
| F1 | 0.514 | 0.310 (Δ-0.205) |

##### q\_id=277 · Causal · chart|text · **DOUBLE-CITE**

**Question:** How did the subscriber base and ARPU of Telkomsel change from 2013 to 2014, and what might have contributed to these changes?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image1', 'image5', 'image7', 'text7'] | ['image1', 'image1_text', 'image5', 'image5_text', 'image7', 'image7_text', 'text7'] |
| retrieved | `image2`, `text11`, `image1` **[GOLD]**, `image7` **[GOLD]**, `image6`, `image3`, `text8`, `text5`, `text1`, `text9` | `image2`, `image1_text` **[GOLD,TWIN]**, `image6_text` **[TWIN]**, `text11`, `image2_text` **[TWIN]**, `image7_text` **[GOLD,TWIN]**, `image1` **[GOLD]**, `image5_text` **[GOLD,TWIN]**, `image7` **[GOLD]**, `text8` |
| prediction | The subscriber base of Telkomsel increased from 132.7 million in 2013 to 139.3 million in 2014, while the ARPU decreased from 34.6 to 35.4. The increase in the subscriber base could be attributed to the growing demand for mobile services, while the d… | From 2013 to 2014, the subscriber base of Telkomsel increased by 13.4 million, growing from 139.3 million to 152.7 million. The ARPU of Telkomsel also saw a slight increase, rising from 34.6 thousand to 35.4 thousand. These changes could be attribute… |
| F1 | 0.505 | 0.286 (Δ-0.220) |

##### q\_id=39 · Comparative · chart|text · **DOUBLE-CITE**

**Question:** How have the political orientations of Republican and Democratic voters shifted from 2008 to 2016?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image4', 'image8', 'text11', 'text4', 'text9'] | ['image4', 'image4_text', 'image8', 'image8_text', 'text11', 'text4', 'text9'] |
| retrieved | `image2`, `text11` **[GOLD]**, `text3`, `image4` **[GOLD]**, `image8` **[GOLD]**, `text9` **[GOLD]**, `text1`, `image7`, `text5`, `text6` | `image2`, `text11` **[GOLD]**, `image2_text` **[TWIN]**, `image4_text` **[GOLD,TWIN]**, `image8_text` **[GOLD,TWIN]**, `text3`, `image4` **[GOLD]**, `image8` **[GOLD]**, `text9` **[GOLD]**, `text1` |
| prediction | The political orientations of Republican and Democratic voters have shifted from 2008 to 2016. The Republican voters have become more conservative, while the Democratic voters have become more liberal. | From 2008 to 2016, there has been a shift in the political orientations of both Republican and Democratic voters. For Republican voters, the percentage who prefer a more conservative direction has remained relatively stable, with 60% wanting the part… |
| F1 | 0.381 | 0.109 (Δ-0.272) |

##### q\_id=25 · Descriptive · figure · **DOUBLE-CITE**

**Question:** List all countries, except U.S. and German, that are mentioned in some illustrated surveys of this report. Note that we do not count EU as a single country.

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image2', 'image5', 'image6'] | ['image2', 'image2_text', 'image5', 'image5_text', 'image6', 'image6_text'] |
| retrieved | `image7`, `text12`, `text8`, `image5` **[GOLD]**, `image6` **[GOLD]**, `image4`, `text3`, `text11`, `text4`, `image8` | `image7`, `image2_text` **[GOLD,TWIN]**, `image5` **[GOLD]**, `image6` **[GOLD]**, `image4`, `image5_text` **[GOLD,TWIN]**, `image7_text` **[TWIN]**, `image6_text` **[GOLD,TWIN]**, `text12`, `text8` |
| prediction | France, Japan, China, Russia, UK, Canada, Israel, Mexico, Austria | Based on the provided information, the countries mentioned in the surveys, excluding the U.S. and Germany, are:  1. China 2. France 3. Japan 4. Russia 5. UK 6. Canada 7. Israel 8. Mexico |
| F1 | 1.000 | 0.421 (Δ-0.579) |

##### q\_id=174 · Inferential · figure|text · **DOUBLE-CITE**

**Question:** How do perceptions of social media content and behavior differ between men and women, and what are the implications for how social media platforms can tailor their recommendations or advertisements based on these differences?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image2', 'image4', 'text3', 'text6'] | ['image2', 'image2_text', 'image4', 'image4_text', 'text3', 'text6'] |
| retrieved | `image4` **[GOLD]**, `text3` **[GOLD]**, `image8`, `text6` **[GOLD]**, `image2` **[GOLD]**, `image3`, `text1`, `text7`, `text5`, `text8` | `image4` **[GOLD]**, `text3` **[GOLD]**, `image8`, `image4_text` **[GOLD,TWIN]**, `text6` **[GOLD]**, `text1`, `text7`, `image2` **[GOLD]**, `image3`, `text5` |
| prediction | The provided data indicates that men are slightly more likely than women to encounter harassing or abusive behavior online, particularly in terms of seeing people being mean or bullying on social media. Conversely, women are more likely to see kind o… | Perceptions of social media content and behavior vary between men and women, with men more likely to encounter bullying and deception while women tend to see more kindness and support. These differences suggest that social media platforms should cons… |
| F1 | 0.522 | 0.419 (Δ-0.103) |

---

## Pipeline B

### Change summary

| Variant pair | n\_changed | n\_improved (ΔF1>0.1) | n\_degraded (ΔF1<−0.1) | n\_equiv | n\_double\_cite |
|---|---|---|---|---|---|
| original → gold_redundant | 1413 | 359 | 209 | 845 | 504 |
| original → negative_redundant | 1269 | 192 | 243 | 834 | 493 |
| original → mixed_redundant | 1573 | 356 | 248 | 969 | 731 |

### original → gold_redundant

#### Improved cases (359 total, 5 shown)

##### q\_id=324 · Analytical · figure|table|text

**Question:** Which method shows the best performance for news recommendation based on AUC and nDCG@10 metrics?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image4', 'image8', 'text1', 'text2', 'text6'] | ['image4', 'image4_text', 'image8', 'image8_text', 'text1', 'text2', 'text6'] |
| retrieved | `image6`, `image8` **[GOLD]**, `image2`, `image1`, `image3`, `image7`, `text5`, `text7`, `text3`, `text1` **[GOLD]** | `image6`, `image8` **[GOLD]**, `image2`, `image1`, `image4_text` **[GOLD,TWIN]**, `image3`, `image7`, `text5`, `text7`, `text3` |
| prediction | The LSTUR-con method shows the best performance for news recommendation based on AUC and nDCG@10 metrics. | LSTUR-ini shows the best performance for news recommendation based on AUC and nDCG@10 metrics. |
| F1 | 0.333 | 0.471 (Δ+0.137) |

##### q\_id=541 · Causal · table|text

**Question:** How did Shell Midstream Partners, L.P.'s operating income and cash from investing activities change from 2019 to 2020, and what might be the reasons for these changes?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image2', 'image6', 'text11', 'text2', 'text3'] | ['image2', 'image2_text', 'image6', 'image6_text', 'text11', 'text2', 'text3'] |
| retrieved | `text10`, `text3` **[GOLD]**, `text5`, `text9`, `text1`, `text4`, `text2` **[GOLD]**, `text7`, `text8`, `image1` | `text10`, `text3` **[GOLD]**, `image6_text` **[GOLD,TWIN]**, `text5`, `text9`, `text1`, `text4`, `image2_text` **[GOLD,TWIN]**, `text2` **[GOLD]**, `text7` |
| prediction | Operating income decreased by $10 million in 2020 compared to the previous year, while cash from investing activities increased. The decrease was primarily due to lower transportation services revenue and higher operating costs related to planned tur… | Operating income decreased from $597 million in 2019 to $643.8 million, and cash provided by investing activities increased significantly due mainly the decrease of capital expenditures compared with previous years. |
| F1 | 0.255 | 0.390 (Δ+0.135) |

##### q\_id=69 · Comparative · chart|table|text

**Question:** How do the priorities for Trump's presidency differ between Trump and Clinton voters, and what does this suggest about their views on his leadership?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image2', 'image5', 'text10', 'text6', 'text9'] | ['image2', 'image2_text', 'image5', 'image5_text', 'text10', 'text6', 'text9'] |
| retrieved | `image7`, `image4`, `image3`, `image1`, `image8`, `image6`, `image2` **[GOLD]**, `text1`, `text9` **[GOLD]**, `text4` | `image7`, `image4`, `image3`, `image1`, `image8`, `image6`, `image5_text` **[GOLD,TWIN]**, `image2` **[GOLD]**, `text1`, `text9` **[GOLD]** |
| prediction | Trump voters are more likely to name health care as Trump’s first priority (29% vs. 14%), while Clinton supporters prioritize the economy and terrorism over other issues, suggesting that they view his leadership through a different lens than do most … | Trump voters are more likely to name health care as Trump’s first priority, while Clinton supporters focus on the economy. This suggests that there is a significant difference in priorities between these two groups of people and indicates differing v… |
| F1 | 0.282 | 0.416 (Δ+0.134) |

##### q\_id=25 · Descriptive · figure · **DOUBLE-CITE**

**Question:** List all countries, except U.S. and German, that are mentioned in some illustrated surveys of this report. Note that we do not count EU as a single country.

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image2', 'image5', 'image6'] | ['image2', 'image2_text', 'image5', 'image5_text', 'image6', 'image6_text'] |
| retrieved | `image6` **[GOLD]**, `image8`, `image5` **[GOLD]**, `image7`, `image4`, `image2` **[GOLD]**, `image3`, `image1`, `text11`, `text10` | `image6` **[GOLD]**, `image8`, `image5` **[GOLD]**, `image7`, `image4`, `image2` **[GOLD]**, `image3`, `image5_text` **[GOLD,TWIN]**, `image1`, `text11` |
| prediction | UK, China | UK, France, China |
| F1 | 0.364 | 0.500 (Δ+0.136) |

##### q\_id=1145 · Inferential · chart · **DOUBLE-CITE**

**Question:** Which country shows the highest percentage of uncertainty ('Don't know') about entrepreneurship among the options given?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image8'] | ['image8', 'image8_text'] |
| retrieved | `image8` **[GOLD]**, `image6`, `image2`, `image1`, `image7`, `image3`, `image5`, `image4`, `text5`, `text8` | `image8` **[GOLD]**, `image6`, `image2`, `image1`, `image7`, `image3`, `image5`, `image4`, `image8_text` **[GOLD,TWIN]**, `text5` |
| prediction | Libya | Palestine shows the highest percentage of uncertainty ('Don't know') about entrepreneurship among all countries, with 21% indicating they don’t have a clear answer. |
| F1 | 0.000 | 0.296 (Δ+0.296) |

#### Degraded cases (209 total, 5 shown)

##### q\_id=398 · Analytical · table|text

**Question:** How do the enhancements of BERT-MRC and XLNet models affect their performance on the English CoNLL 2003 and Chinese MSRA datasets?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image1', 'image3', 'text5'] | ['image1', 'image1_text', 'image3', 'image3_text', 'text5'] |
| retrieved | `text5` **[GOLD]**, `text9`, `text8`, `text6`, `text4`, `text10`, `text12`, `text2`, `text1`, `text11` | `text5` **[GOLD]**, `text9`, `image3_text` **[GOLD,TWIN]**, `image1_text` **[GOLD,TWIN]**, `text8`, `text6`, `text4`, `text10`, `text12`, `text2` |
| prediction | The enhancements of BERT-MRC and XLNet models result in significant performance boosts on both English CoNLL 2003 (87.65 EM, F1: +$+9\,\mathrm{F}1$) as well Chinese MSRA datasets | The enhancements of BERT-MRC and XLNet models improve their performance on the English CoNLL 2003 dataset. The F1 scores for all three metrics (Precision, Recall) increase with each enhancement applied to these two datasets.  On Chinese MSRA data set… |
| F1 | 0.582 | 0.436 (Δ-0.146) |

##### q\_id=75 · Causal · chart|table|text · **DOUBLE-CITE**

**Question:** How do voter perceptions of political entities and campaign negativity in the 2016 election relate to each other?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image1', 'image4', 'text12', 'text2'] | ['image1', 'image1_text', 'image4', 'image4_text', 'text12', 'text2'] |
| retrieved | `image3`, `image7`, `image1` **[GOLD]**, `image6`, `text10`, `text4`, `text12` **[GOLD]**, `text2` **[GOLD]**, `image2`, `text1` | `image3`, `image7`, `image1` **[GOLD]**, `image6`, `text10`, `text4`, `text12` **[GOLD]**, `image1_text` **[GOLD,TWIN]**, `text2` **[GOLD]**, `image2` |
| prediction | Voter perceptions of political entities and campaign negativity in the 2016 election are related. | Voter perceptions of political entities and campaign negativity in the 2016 election are related, with a high percentage (94%) saying there was more mudslinging or negative campaigning than past elections. This perception is shared by Clinton voters … |
| F1 | 0.545 | 0.384 (Δ-0.162) |

##### q\_id=151 · Comparative · chart|table|text

**Question:** How do internet and smartphone adoption rates among older adults (65+) vary by income and education, and how do these rates compare to overall trends in device ownership?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image1', 'image3', 'image4', 'image6', 'text1', 'text4', 'text8'] | ['image1', 'image1_text', 'image3', 'image3_text', 'image4', 'image4_text', 'image6', 'image6_text', 'text1', 'text4', 'text8'] |
| retrieved | `image5`, `image3` **[GOLD]**, `image8`, `image2`, `image7`, `text1` **[GOLD]**, `text10`, `text9`, `text11`, `text7` | `image5`, `image3` **[GOLD]**, `image8`, `image2`, `image7`, `text1` **[GOLD]**, `text10`, `text9`, `image1_text` **[GOLD,TWIN]**, `text11` |
| prediction | The internet and smartphone adoption rates among older adults (65+) vary by income, with affluent seniors adopting these technologies at higher levels than their lower-income counterparts. The data also shows that the overall trends in device ownersh… | Overall, internet and smartphone adoption rates among older adults (65+) vary by income level. Seniors with higher incomes are more likely to have both an online presence as well broadband at home compared than those in lower-income brackets ($30K-$4… |
| F1 | 0.333 | 0.202 (Δ-0.132) |

##### q\_id=31 · Descriptive · figure · **DOUBLE-CITE**

**Question:** How many figures are related to Airbus?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image4', 'image5', 'image6', 'image7', 'image8'] | ['image4', 'image4_text', 'image5', 'image5_text', 'image6', 'image6_text', 'image7', 'image7_text', 'image8', 'image8_text'] |
| retrieved | `image8` **[GOLD]**, `text6`, `text1`, `text11`, `image6` **[GOLD]**, `text5`, `text7`, `text9`, `text10`, `text2` | `image8` **[GOLD]**, `text6`, `text1`, `image7_text` **[GOLD,TWIN]**, `image4_text` **[GOLD,TWIN]**, `text11`, `image6` **[GOLD]**, `image8_text` **[GOLD,TWIN]**, `image6_text` **[GOLD,TWIN]**, `text5` |
| prediction | There are 5 figures related to Airbus. | There are 6 figures related to Airbus. |
| F1 | 0.250 | 0.000 (Δ-0.250) |

##### q\_id=174 · Inferential · figure|text · **DOUBLE-CITE**

**Question:** How do perceptions of social media content and behavior differ between men and women, and what are the implications for how social media platforms can tailor their recommendations or advertisements based on these differences?

| | original | gold_redundant |
|---|---|---|
| gold\_set | ['image2', 'image4', 'text3', 'text6'] | ['image2', 'image2_text', 'image4', 'image4_text', 'text3', 'text6'] |
| retrieved | `image4` **[GOLD]**, `image8`, `image3`, `text10`, `text6` **[GOLD]**, `image2` **[GOLD]**, `image6`, `text9`, `text4`, `text7` | `image4` **[GOLD]**, `image8`, `image3`, `image4_text` **[GOLD,TWIN]**, `text10`, `text6` **[GOLD]**, `image2` **[GOLD]**, `image6`, `text9`, `text4` |
| prediction | Perceptions of social media content and behavior differ between men (29%)and women(19%), with a 30-point gender gap. Men are somewhat more likely than woman to see people being mean or bullying, deceptive on the internet. Social platforms can tailor … | Perceptions of social media content and behavior differ between men (29%)and women, with 17%of each gender saying people are kind or supportive. Men were more likely to say they see mean/bullying online behaviors than did their female counterparts: W… |
| F1 | 0.435 | 0.320 (Δ-0.115) |

### original → negative_redundant

#### Improved cases (192 total, 5 shown)

##### q\_id=431 · Analytical · figure|table|text · **DOUBLE-CITE**

**Question:** How does the DyGIE model's performance on entity and relation extraction vary across different datasets and configurations, and what role do the CorefProp and RelProp components play in these variations?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image1', 'image2', 'image3', 'image4', 'image8', 'text7'] | ['image1', 'image2', 'image3', 'image4', 'image8', 'text7'] |
| retrieved | `image6`, `image2` **[GOLD]**, `text7` **[GOLD]**, `text8`, `text5`, `text9`, `text1`, `text10`, `text2`, `text12` | `image6`, `image2` **[GOLD]**, `text7` **[GOLD]**, `text8`, `text5`, `image6_text` **[TWIN]**, `text9`, `text1`, `image5_text` **[TWIN]**, `text10` |
| prediction | The DyGIE model's performance on entity and relation extraction varies across different datasets, with the addition of CorefProp providing a small benefit in sentences containing more entities. In SciERC dataset where pronouns are uniformly assigned … | The DyGIE model's performance on entity and relation extraction varies across different datasets, with the best results observed in ACE05 (Entity score of 87.4 for entities; Relation F1 scores between approximately 62-63) where both Coreference Propa… |
| F1 | 0.273 | 0.395 (Δ+0.123) |

##### q\_id=576 · Causal · table|text

**Question:** How did the number of individuals served by UnitedHealthcare change across different segments from 2019 to 2020, and what might be the causes of these changes?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image4', 'text10', 'text5'] | ['image4', 'text10', 'text5'] |
| retrieved | `text4`, `text5` **[GOLD]**, `text10` **[GOLD]**, `text3`, `text7`, `text6`, `text1`, `text11`, `text8`, `text2` | `image2_text` **[TWIN]**, `text4`, `text5` **[GOLD]**, `text10` **[GOLD]**, `text3`, `text7`, `text6`, `text1`, `text11`, `text8` |
| prediction | The number of individuals served by UnitedHealthcare increased in Medicare Advantage and Medicaid, while decreasing for commercial businesses (including Global) due to factors such as COVID-19-related care deferrals. The increase was also driven part… | The number of individuals served by UnitedHealthcare increased in the Medicare Advantage and Medicaid segments, while it decreased for commercial business. The increase was primarily due to growth through individual plans within Medicare Advantage as… |
| F1 | 0.156 | 0.304 (Δ+0.148) |

##### q\_id=81 · Comparative · table|text · **DOUBLE-CITE**

**Question:** How do different age groups perceive the government's performance in reducing the terrorist threat in 2015, and how does this compare with their views on whether anti-terror policies have gone too far in restricting civil liberties or not far enough to protect the country?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image2', 'image5', 'text6'] | ['image2', 'image5', 'text6'] |
| retrieved | `image3`, `image6`, `image4`, `image7`, `text11`, `text4`, `text2`, `text3`, `text12`, `text6` **[GOLD]** | `image3`, `image6`, `image6_text` **[TWIN]**, `image4`, `image7`, `text11`, `text4`, `text2`, `text3`, `text12` |
| prediction | Older Americans are more likely to give the government low marks for reducing terrorism, and they also express greater concern that anti-terrorism policies have not gone far enough. | Older Americans are more likely to give the government low marks for reducing terrorism threats, and they also tend not be as concerned that anti-terrorism policies have gone too far in restricting civil liberties. |
| F1 | 0.207 | 0.344 (Δ+0.137) |

##### q\_id=51 · Descriptive · table

**Question:** Which country had new cities added to the survey in 2013 and what were those cities?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image1', 'image8'] | ['image1', 'image8'] |
| retrieved | `image5`, `image7`, `image8` **[GOLD]**, `text2`, `text3`, `text6`, `text12`, `text4`, `text11`, `image3` | `image5`, `image7`, `image8` **[GOLD]**, `text2`, `text3`, `text6`, `text12`, `text4`, `text11`, `image4_text` **[TWIN]** |
| prediction | In 2013, Syria was added to the survey. The cities included were Damascus and Aleppo (the latter is also mentioned in a later year). | The country that had new cities added to the survey in 2013 was Lebanon, and those were Beirut (60%), Saida(25%) , Tripoli. |
| F1 | 0.323 | 0.467 (Δ+0.144) |

##### q\_id=591 · Inferential · table|text

**Question:** How did the total loans and deposits change across different lines of business from 2020 to 2021, and what were the contributing factors?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image3', 'image6', 'text11', 'text12', 'text8', 'text9'] | ['image3', 'image6', 'text11', 'text12', 'text8', 'text9'] |
| retrieved | `text12` **[GOLD]**, `text9` **[GOLD]**, `text10`, `text5`, `text8` **[GOLD]**, `text7`, `text3`, `text11` **[GOLD]**, `text6`, `text2` | `text12` **[GOLD]**, `text9` **[GOLD]**, `image4_text` **[TWIN]**, `image7_text` **[TWIN]**, `text10`, `text5`, `text8` **[GOLD]**, `image8_text` **[TWIN]**, `text7`, `image2_text` **[TWIN]** |
| prediction | From 2020 to 2021, total loans decreased due to lower loan demand and higher paydowns. However, there was modest growth in late-2021 driven by increased line utilization and customer growth.  Total deposits (average) also saw an increase during this … | Total loans (average) decreased in 2021 driven by lower loan demand, including a decrease of line utilization and higher paydowns. Total deposits increased due to government stimulus programs for consumer customers as well as economic uncertainty ass… |
| F1 | 0.216 | 0.436 (Δ+0.220) |

#### Degraded cases (243 total, 5 shown)

##### q\_id=401 · Analytical · table|text

**Question:** How do different data augmentation techniques impact the performance of BERT models on the QOP dataset, and how is this effect measured across various sentiment analysis and named entity recognition tasks?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image2', 'image4', 'text10', 'text2', 'text9'] | ['image2', 'image4', 'text10', 'text2', 'text9'] |
| retrieved | `text4`, `text12`, `text11`, `text2` **[GOLD]**, `text9` **[GOLD]**, `text5`, `text7`, `text6`, `text1`, `text10` **[GOLD]** | `image6_text` **[TWIN]**, `image3_text` **[TWIN]**, `image5_text` **[TWIN]**, `text4`, `text12`, `image7_text` **[TWIN]**, `text11`, `image1_text` **[TWIN]**, `image8_text` **[TWIN]**, `text2` **[GOLD]** |
| prediction | Different data augmentation techniques impact the performance of BERT models on QQP dataset, and this effect is measured across various sentiment analysis (SST-2) named entity recognition tasks. | Different data augmentation techniques impact the performance of BERT models on various sentiment analysis and named entity recognition tasks. The effect is measured by comparing F1 scores for different variants, such as +FL (small improvement), DL (… |
| F1 | 0.523 | 0.329 (Δ-0.194) |

##### q\_id=1591 · Application-based · figure|text · **DOUBLE-CITE**

**Question:** Which areas do candidates need to focus on for the LinkedIn Recruiter Certification exam, and how does understanding Boolean search results through a Venn diagram relate to these topic areas?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image7', 'text4'] | ['image7', 'text4'] |
| retrieved | `image4`, `image3`, `image7` **[GOLD]**, `text10`, `image1`, `text4` **[GOLD]**, `text11`, `text5`, `text6`, `text2` | `image4`, `image3`, `image7` **[GOLD]**, `text10`, `image1`, `image4_text` **[TWIN]**, `text4` **[GOLD]**, `text11`, `text5`, `text6` |
| prediction | Candidates need to focus on the entire recruiting life cycle, including building a talent pipeline and posting jobs. Understanding Boolean search results through Venn diagrams relates as it is an important skill for all recruiters in efficiently sear… | Candidates need to focus on the following areas for LinkedIn Recruiter Certification: identifying talent through search, engaging with potential candidates using their presence and InMail features of LinkedIn; building a pipeline by creating an effec… |
| F1 | 0.513 | 0.378 (Δ-0.134) |

##### q\_id=544 · Causal · table|text

**Question:** How did the free cash flow change from 2020 to 2021, and what were the main contributing factors according to the financial statements?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image5', 'image8', 'text3', 'text5', 'text6', 'text9'] | ['image5', 'image8', 'text3', 'text5', 'text6', 'text9'] |
| retrieved | `text8`, `text10`, `text5` **[GOLD]**, `text3` **[GOLD]**, `text12`, `text9` **[GOLD]**, `text7`, `text6` **[GOLD]**, `text2`, `text4` | `image6_text` **[TWIN]**, `image2_text` **[TWIN]**, `text8`, `text10`, `text5` **[GOLD]**, `text3` **[GOLD]**, `text12`, `text9` **[GOLD]**, `text7`, `text6` **[GOLD]** |
| prediction | Free cash flow increased by USD 1.6 billion from 2020 to 2021, mainly driven by higher operating income adjusted for non-cash items and lower payments out of provisions due to legal matters in the prior year; however it was partly offset by a upfront… | Free cash flow increased to USD 13.3 billion in the year ended December 20, compared with $USD (+)$+9\%) of free cash flows for fiscal years ending on Decembe r-December -86; and (Decemeber-Decemebr ) respectively. The increase was mainly driven by h… |
| F1 | 0.518 | 0.342 (Δ-0.175) |

##### q\_id=46 · Comparative · figure|text · **DOUBLE-CITE**

**Question:** How did the percentages for Republicans and Democrats wanting their leaders to 'stand up' to opposition change from January 2018 to January 2019?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image7', 'image8', 'text1'] | ['image7', 'image8', 'text1'] |
| retrieved | `image5`, `image3`, `image8` **[GOLD]**, `image7` **[GOLD]**, `image6`, `image2`, `image1`, `image4`, `text6`, `text9` | `image5`, `image3`, `image8` **[GOLD]**, `image7` **[GOLD]**, `image6`, `image2`, `image1`, `image4`, `image5_text` **[TWIN]**, `image1_text` **[TWIN]** |
| prediction | The percentage of Republicans wanting their leaders to 'stand up' increased from 40% in January 2018, while the Democrats wanted this by a margin that grew slightly more than two-to-one. | The percentage of Republicans wanting their leaders to "stand up" more against opposition increased from 38% in January 2017, dropped slightly but then rose again by the end of that year. In contrast, Democrats' desire for a stronger stance on opposi… |
| F1 | 0.279 | 0.138 (Δ-0.141) |

##### q\_id=303 · Descriptive · figure

**Question:** Which figures include line plots in the paper?

| | original | negative_redundant |
|---|---|---|
| gold\_set | ['image2', 'image3'] | ['image2', 'image3'] |
| retrieved | `text10`, `text2`, `text8`, `text3`, `text4`, `text12`, `text6`, `text9`, `text7`, `text5` | `image5_text` **[TWIN]**, `text10`, `text2`, `text8`, `text3`, `text4`, `text12`, `image4_text` **[TWIN]**, `text6`, `text9` |
| prediction | Figures 5a, Figure 6c and figure of the paper. | Figures 6a and Figure 5b include line plots in the paper. |
| F1 | 0.333 | 0.143 (Δ-0.190) |

### original → mixed_redundant

#### Improved cases (356 total, 5 shown)

##### q\_id=323 · Analytical · table|text · **DOUBLE-CITE**

**Question:** Which model achieves the highest F1-value for Named Entity Recognition (NER) and what features contribute to it?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image3', 'image4', 'text9'] | ['image3', 'image3_text', 'image4', 'image4_text', 'text9'] |
| retrieved | `image1`, `image5`, `text5`, `image2`, `text7`, `text8`, `text9` **[GOLD]**, `text1`, `text11`, `text12` | `image1`, `image4_text` **[GOLD,TWIN]**, `image5`, `text5`, `image3_text` **[GOLD,TWIN]**, `image2`, `text7`, `text8`, `text9` **[GOLD]**, `image5_text` **[TWIN]** |
| prediction | The model that achieves the highest F1-value for Named Entity Recognition (NER) is "CCNN $^+$ WLSTM  $ ^+   CRF$". The features contributing to it are character sequence and word-sequence representations, with LSTM or CNN used in both cases. | The model "CCNN+WLSTM+CRF" achieves the highest F1-value for Named Entity Recognition (NER) with a score of 91.35, and it contributes to this performance by using automatic features such as CCNN encoding character sequence information combined with W… |
| F1 | 0.392 | 0.523 (Δ+0.131) |

##### q\_id=1995 · Application-based · table · **DOUBLE-CITE**

**Question:** How do the different metering modes and focus modes in Pro Mode enhance photography under various scenarios?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image4', 'image5'] | ['image4', 'image4_text', 'image5', 'image5_text'] |
| retrieved | `image3`, `text8`, `text3`, `text9`, `text6`, `image2`, `text11`, `text1`, `text4`, `text10` | `image3`, `text8`, `text3`, `image5_text` **[GOLD,TWIN]**, `text9`, `image4_text` **[GOLD,TWIN]**, `image3_text` **[TWIN]**, `text6`, `image2`, `text11` |
| prediction | The different metering modes and focus modes in Pro Mode enhance photography under various scenarios by allowing users to fine-tune photos, giving full control over ISO sensitivity for better exposure settings. The ability of the camera app's flash f… | The different metering modes and focus modes in Pro Mode enhance photography by allowing users to fine-tune camera settings for specific scenarios. For instance, matrix or center-metered mode is ideal when shooting landscapes with varying light condi… |
| F1 | 0.188 | 0.337 (Δ+0.148) |

##### q\_id=613 · Causal · table|text

**Question:** What are some potential impacts and mitigations for supply chain disruptions as described in the report, and how do these relate to Nestlé's factory distribution across different regions?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image1', 'image3', 'image5', 'text1', 'text10', 'text12', 'text2'] | ['image1', 'image1_text', 'image3', 'image3_text', 'image5', 'image5_text', 'text1', 'text10', 'text12', 'text2'] |
| retrieved | `image2`, `text6`, `text1` **[GOLD]**, `text10` **[GOLD]**, `text8`, `text2` **[GOLD]**, `text12` **[GOLD]**, `image4`, `text5`, `text4` | `image6_text` **[TWIN]**, `image2`, `text6`, `text1` **[GOLD]**, `text10` **[GOLD]**, `text8`, `text2` **[GOLD]**, `text12` **[GOLD]**, `image4`, `text5` |
| prediction | Nestlé is working with supply chain partners to pilot solutions that will better balance efficiency and resiliency. For example, in 2020 we increased the scope of our Transport Hub technologies... | Supply chain disruptions can have a significant impact on Nestlé's operations, including operational disruption and financial losses. To mitigate these risks, the company has implemented various strategies such as developing business continuity plans… |
| F1 | 0.127 | 0.264 (Δ+0.137) |

##### q\_id=69 · Comparative · chart|table|text

**Question:** How do the priorities for Trump's presidency differ between Trump and Clinton voters, and what does this suggest about their views on his leadership?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image2', 'image5', 'text10', 'text6', 'text9'] | ['image2', 'image2_text', 'image5', 'image5_text', 'text10', 'text6', 'text9'] |
| retrieved | `image7`, `image4`, `image3`, `image1`, `image8`, `image6`, `image2` **[GOLD]**, `text1`, `text9` **[GOLD]**, `text4` | `image7`, `image4`, `image3`, `image1`, `image8`, `image6`, `image5_text` **[GOLD,TWIN]**, `image2` **[GOLD]**, `text1`, `text9` **[GOLD]** |
| prediction | Trump voters are more likely to name health care as Trump’s first priority (29% vs. 14%), while Clinton supporters prioritize the economy and terrorism over other issues, suggesting that they view his leadership through a different lens than do most … | Trump voters are more likely to name health care as Trump’s first priority, while Clinton supporters focus on the economy. This suggests that there is a significant difference in priorities between these two groups of people and indicates differing v… |
| F1 | 0.282 | 0.416 (Δ+0.134) |

##### q\_id=7 · Descriptive · figure

**Question:** How many line plots are in this report?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image3', 'image4', 'image7'] | ['image3', 'image3_text', 'image4', 'image4_text', 'image7', 'image7_text'] |
| retrieved | `text8`, `text2`, `text12`, `text11`, `text9`, `text1`, `text6`, `text10`, `text3`, `text5` | `image3_text` **[GOLD,TWIN]**, `image4_text` **[GOLD,TWIN]**, `image6_text` **[TWIN]**, `image7_text` **[GOLD,TWIN]**, `text8`, `text2`, `image1_text` **[TWIN]**, `text12`, `image8_text` **[TWIN]**, `text11` |
| prediction | There are no line plots mentioned in this report. | There are 6 line plots in this report. |
| F1 | 0.000 | 0.222 (Δ+0.222) |

#### Degraded cases (248 total, 5 shown)

##### q\_id=401 · Analytical · table|text

**Question:** How do different data augmentation techniques impact the performance of BERT models on the QOP dataset, and how is this effect measured across various sentiment analysis and named entity recognition tasks?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image2', 'image4', 'text10', 'text2', 'text9'] | ['image2', 'image2_text', 'image4', 'image4_text', 'text10', 'text2', 'text9'] |
| retrieved | `text4`, `text12`, `text11`, `text2` **[GOLD]**, `text9` **[GOLD]**, `text5`, `text7`, `text6`, `text1`, `text10` **[GOLD]** | `image6_text` **[TWIN]**, `image3_text` **[TWIN]**, `image5_text` **[TWIN]**, `image2_text` **[GOLD,TWIN]**, `text4`, `text12`, `image4_text` **[GOLD,TWIN]**, `image7_text` **[TWIN]**, `text11`, `image1_text` **[TWIN]** |
| prediction | Different data augmentation techniques impact the performance of BERT models on QQP dataset, and this effect is measured across various sentiment analysis (SST-2) named entity recognition tasks. | Different data augmentation techniques impact the performance of BERT models on QQP and SST-2 datasets, as shown in Table 1. The effect is measured by F1 scores for MRPC (Multilingual Recurrent Pre-trained Transformer) tasks.  BERT+FL shows a small i… |
| F1 | 0.523 | 0.295 (Δ-0.229) |

##### q\_id=1591 · Application-based · figure|text · **DOUBLE-CITE**

**Question:** Which areas do candidates need to focus on for the LinkedIn Recruiter Certification exam, and how does understanding Boolean search results through a Venn diagram relate to these topic areas?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image7', 'text4'] | ['image7', 'image7_text', 'text4'] |
| retrieved | `image4`, `image3`, `image7` **[GOLD]**, `text10`, `image1`, `text4` **[GOLD]**, `text11`, `text5`, `text6`, `text2` | `image4`, `image3`, `image7` **[GOLD]**, `text10`, `image1`, `image4_text` **[TWIN]**, `text4` **[GOLD]**, `text11`, `text5`, `text6` |
| prediction | Candidates need to focus on the entire recruiting life cycle, including building a talent pipeline and posting jobs. Understanding Boolean search results through Venn diagrams relates as it is an important skill for all recruiters in efficiently sear… | Candidates need to focus on the following areas for LinkedIn Recruiter Certification: identifying talent through search, engaging with potential candidates using their presence and InMail features of LinkedIn; building a pipeline by creating an effec… |
| F1 | 0.513 | 0.378 (Δ-0.134) |

##### q\_id=75 · Causal · chart|table|text · **DOUBLE-CITE**

**Question:** How do voter perceptions of political entities and campaign negativity in the 2016 election relate to each other?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image1', 'image4', 'text12', 'text2'] | ['image1', 'image1_text', 'image4', 'image4_text', 'text12', 'text2'] |
| retrieved | `image3`, `image7`, `image1` **[GOLD]**, `image6`, `text10`, `text4`, `text12` **[GOLD]**, `text2` **[GOLD]**, `image2`, `text1` | `image3`, `image7`, `image1` **[GOLD]**, `image6`, `text10`, `text4`, `text12` **[GOLD]**, `image7_text` **[TWIN]**, `image1_text` **[GOLD,TWIN]**, `text2` **[GOLD]** |
| prediction | Voter perceptions of political entities and campaign negativity in the 2016 election are related. | Voter perceptions of political entities and campaign negativity in the 2016 election are related. The graph shows that a high percentage (92%) said there was more mudslinging or negative campaigning than past elections, which is higher compared to pr… |
| F1 | 0.545 | 0.337 (Δ-0.208) |

##### q\_id=46 · Comparative · figure|text · **DOUBLE-CITE**

**Question:** How did the percentages for Republicans and Democrats wanting their leaders to 'stand up' to opposition change from January 2018 to January 2019?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image7', 'image8', 'text1'] | ['image7', 'image7_text', 'image8', 'image8_text', 'text1'] |
| retrieved | `image5`, `image3`, `image8` **[GOLD]**, `image7` **[GOLD]**, `image6`, `image2`, `image1`, `image4`, `text6`, `text9` | `image5`, `image3`, `image8` **[GOLD]**, `image7` **[GOLD]**, `image6`, `image2`, `image1`, `image4`, `image5_text` **[TWIN]**, `image1_text` **[TWIN]** |
| prediction | The percentage of Republicans wanting their leaders to 'stand up' increased from 40% in January 2018, while the Democrats wanted this by a margin that grew slightly more than two-to-one. | The percentage of Republicans wanting their leaders to "stand up" more against opposition increased from 38% in January 2017, dropped slightly but then rose again by the end of that year. In contrast, Democrats' desire for a stronger stance on opposi… |
| F1 | 0.279 | 0.138 (Δ-0.141) |

##### q\_id=31 · Descriptive · figure · **DOUBLE-CITE**

**Question:** How many figures are related to Airbus?

| | original | mixed_redundant |
|---|---|---|
| gold\_set | ['image4', 'image5', 'image6', 'image7', 'image8'] | ['image4', 'image4_text', 'image5', 'image5_text', 'image6', 'image6_text', 'image7', 'image7_text', 'image8', 'image8_text'] |
| retrieved | `image8` **[GOLD]**, `text6`, `text1`, `text11`, `image6` **[GOLD]**, `text5`, `text7`, `text9`, `text10`, `text2` | `image8` **[GOLD]**, `text6`, `text1`, `image7_text` **[GOLD,TWIN]**, `image4_text` **[GOLD,TWIN]**, `text11`, `image6` **[GOLD]**, `image8_text` **[GOLD,TWIN]**, `image6_text` **[GOLD,TWIN]**, `text5` |
| prediction | There are 5 figures related to Airbus. | There are 6 figures related to Airbus. |
| F1 | 0.250 | 0.000 (Δ-0.250) |

---
