# When Does Redundancy Happen?

> **TL;DR.** Across both pipelines and all three redundancy variants, twins appear in top-10 in 81% of questions on average (highest: A mixed_redundant at 90%). The most common question types among active redundancy cases are Descriptive, Comparative, Interpretative. Redundancy activation rate is similar across evidence modality types, with multi-modal questions (table|text, figure|text) appearing most frequently due to their prevalence in the dataset.

---

## Pipeline A

### Summary

| Variant | n\_active | n\_inactive | mean\_twins (active) | top question\_types |
|---|---|---|---|---|
| gold_redundant | 1669 | 331 | 1.40 | Descriptive=592, Comparative=578, Interpretative=254 |
| negative_redundant | 1529 | 471 | 2.62 | Descriptive=537, Comparative=517, Interpretative=239 |
| mixed_redundant | 1802 | 198 | 3.14 | Descriptive=635, Comparative=621, Interpretative=279 |

### Question-type breakdown (active questions only)

| Question type | gold\_r | neg\_r | mix\_r |
|---|---|---|---|
| Analytical | 173 | 170 | 190 |
| Application-based | 2 | 2 | 2 |
| Causal | 15 | 16 | 16 |
| Comparative | 578 | 517 | 621 |
| Descriptive | 592 | 537 | 635 |
| Inferential | 30 | 28 | 33 |
| Interpretative | 254 | 239 | 279 |
| Procedural | 25 | 20 | 26 |

### Evidence modality breakdown (active questions, top 8)

| Modality | gold\_r | neg\_r | mix\_r |
|---|---|---|---|
| table|text | 514 | 486 | 538 |
| table | 353 | 329 | 360 |
| figure | 221 | 210 | 240 |
| figure|text | 203 | 181 | 235 |
| chart|text | 151 | 118 | 183 |
| chart | 88 | 84 | 91 |
| chart|table|text | 43 | 39 | 49 |
| figure|table|text | 37 | 30 | 39 |

### Representative active cases — gold_redundant

#### q\_id=4 · Analytical · chart

**Question:** Among the top 10 findings, how many of them are supported by statistical results grouped by the countries?

**gold\_set:** ['image3', 'image3_text', 'image4', 'image4_text', 'image6', 'image6_text', 'image7', 'image7_text']

**retrieved (top-10):** `image6` **[GOLD]**, `text5`, `image7` **[GOLD]**, `image4` **[GOLD]**, `image3` **[GOLD]**, `text11`, `image6_text` **[GOLD TWIN]**, `image1`, `image4_text` **[GOLD TWIN]**, `image3_text` **[GOLD TWIN]**

**prediction:** Based on the information provided in the images, there are no direct references to the "top 10 findings" or any specific statistical results grouped by countries that can be directly counted. The images show data on concerns, confidence levels, and responses but do not explicitly list the "top 10 fi…

**answer\_ref:** `4`

#### q\_id=1591 · Application-based · figure|text

**Question:** Which areas do candidates need to focus on for the LinkedIn Recruiter Certification exam, and how does understanding Boolean search results through a Venn diagram relate to these topic areas?

**gold\_set:** ['image7', 'image7_text', 'text4']

**retrieved (top-10):** `image4`, `text4` **[GOLD]**, `text10`, `text11`, `text6`, `image3`, `text5`, `text12`, `image7_text` **[GOLD TWIN]**, `text3`

**prediction:** Candidates need to focus on several key areas for the LinkedIn Recruiter Certification exam, including:  1. **Engaging talent**: Utilizing LinkedIn presence and InMail. 2. **Identifying talent**: Conducting effective searches. 3. **Building a talent pipeline**: Managing and optimizing pipelines. 4. …

**answer\_ref:** `Candidates need to focus on building a talent pipeline, posting jobs, maximizing efficiency, engaging talent, and identifying talent for the LinkedIn Recruiter Certification exam. Understanding Boolean search results, as demonstrated in a Venn diagram, relates to identifying talent using search techniques.`

#### q\_id=75 · Causal · chart|table|text

**Question:** How do voter perceptions of political entities and campaign negativity in the 2016 election relate to each other?

**gold\_set:** ['image1', 'image1_text', 'image4', 'image4_text', 'text12', 'text2']

**retrieved (top-10):** `image3`, `text12` **[GOLD]**, `image1` **[GOLD]**, `text2` **[GOLD]**, `text4`, `image6`, `image4_text` **[GOLD TWIN]**, `text3`, `image1_text` **[GOLD TWIN]**, `text8`

**prediction:** Voter perceptions of political entities and campaign negativity in the 2016 election are closely related. The data shows that a large majority of voters viewed the 2016 campaign as exceptionally negative, with 92% saying there was more mudslinging or negative campaigning than in past elections. This…

**answer\_ref:** `Voter perceptions in the 2016 election were generally negative, with poor grades for political entities and high levels of perceived mudslinging.`

#### q\_id=3 · Comparative · chart|text

**Question:** What percentage of the party that holds the highest total percentage of good policy ideas and high ethical standards reported that they followed the news about congressional elections in their state very closely in the survey of U.S. adults conducted from April 25 to May 1, 2018?

**gold\_set:** ['image4', 'image4_text', 'image7', 'image7_text', 'text2', 'text3', 'text6']

**retrieved (top-10):** `image7` **[GOLD]**, `text2` **[GOLD]**, `image7_text` **[GOLD TWIN]**, `text3` **[GOLD]**, `text5`, `text12`, `text6` **[GOLD]**, `text1`, `text7`, `image5`

**prediction:** Based on the information provided, the Republican Party holds the highest total percentage for both "Has good policy ideas" (50%) and "Has high ethical standards" (41%). According to the survey data, 19% of Americans said they were following news about the candidates and campaign in their state or d…

**answer\_ref:** `21%`

#### q\_id=0 · Descriptive · chart|text

**Question:** In 2015, how many percentage of surveyed adults, Republicans, and Democrats believed that the goverment was doing very/fairly well in reducing the threat of terrorism? Please write the answer in list format, e.g., ["3","2"]

**gold\_set:** ['image4', 'image4_text', 'image7', 'image7_text', 'text6', 'text9']

**retrieved (top-10):** `image5`, `text9` **[GOLD]**, `text1`, `text6` **[GOLD]**, `text4`, `text5`, `text2`, `image4_text` **[GOLD TWIN]**, `text12`, `text3`

**prediction:** ["46", "34"]

**answer\_ref:** `46 27 64`

### Representative active cases — negative_redundant

#### q\_id=4 · Analytical · chart

**Question:** Among the top 10 findings, how many of them are supported by statistical results grouped by the countries?

**gold\_set:** ['image3', 'image4', 'image6', 'image7']

**retrieved (top-10):** `image6` **[GOLD]**, `text5`, `image7` **[GOLD]**, `image4` **[GOLD]**, `image3` **[GOLD]**, `text11`, `image5_text` **[TWIN]**, `image1`, `image1_text` **[TWIN]**, `text1`

**prediction:** The image provided does not contain any statistical results or data grouped by countries that can be directly compared to the "top 10 findings." The image appears to be a bar graph showing levels of concern about various topics across different countries, but it does not provide numerical data or st…

**answer\_ref:** `4`

#### q\_id=1591 · Application-based · figure|text

**Question:** Which areas do candidates need to focus on for the LinkedIn Recruiter Certification exam, and how does understanding Boolean search results through a Venn diagram relate to these topic areas?

**gold\_set:** ['image7', 'text4']

**retrieved (top-10):** `image4`, `image4_text` **[TWIN]**, `image3`, `text4` **[GOLD]**, `text10`, `text11`, `text6`, `image3_text` **[TWIN]**, `text5`, `text12`

**prediction:** Candidates need to focus on several key areas for the LinkedIn Recruiter Certification exam, including identifying talent, engaging talent, building a talent pipeline, posting jobs, and maximizing efficiency. Understanding how to construct accurate Boolean search strings, as represented by a Venn di…

**answer\_ref:** `Candidates need to focus on building a talent pipeline, posting jobs, maximizing efficiency, engaging talent, and identifying talent for the LinkedIn Recruiter Certification exam. Understanding Boolean search results, as demonstrated in a Venn diagram, relates to identifying talent using search techniques.`

#### q\_id=75 · Causal · chart|table|text

**Question:** How do voter perceptions of political entities and campaign negativity in the 2016 election relate to each other?

**gold\_set:** ['image1', 'image4', 'text12', 'text2']

**retrieved (top-10):** `image3`, `text12` **[GOLD]**, `image1` **[GOLD]**, `text2` **[GOLD]**, `text4`, `image6`, `text3`, `text8`, `text1`, `image3_text` **[TWIN]**

**prediction:** Voter perceptions of political entities and campaign negativity in the 2016 election are closely related. The data shows that a significant majority of voters viewed the 2016 campaign as more negative than past elections, with 92% saying there was more mudslinging or negative campaigning compared to…

**answer\_ref:** `Voter perceptions in the 2016 election were generally negative, with poor grades for political entities and high levels of perceived mudslinging.`

#### q\_id=12 · Comparative · table

**Question:** What is the difference in total volume between the rank 1 and rank 19 top albums?

**gold\_set:** ['image2', 'image3']

**retrieved (top-10):** `image4`, `image6_text` **[TWIN]**, `text5`, `image8_text` **[TWIN]**, `image8`, `image5_text` **[TWIN]**, `image5`, `image4_text` **[TWIN]**, `text10`, `text3`

**prediction:** The information provided does not specify the exact volumes for the top 1 and 19 albums. Therefore, it's not possible to calculate the difference in total volume between these two specific ranks based on the given data.

**answer\_ref:** `1194000`

#### q\_id=0 · Descriptive · chart|text

**Question:** In 2015, how many percentage of surveyed adults, Republicans, and Democrats believed that the goverment was doing very/fairly well in reducing the threat of terrorism? Please write the answer in list format, e.g., ["3","2"]

**gold\_set:** ['image4', 'image7', 'text6', 'text9']

**retrieved (top-10):** `image5`, `text9` **[GOLD]**, `text1`, `text6` **[GOLD]**, `text4`, `text5`, `text2`, `image5_text` **[TWIN]**, `image8_text` **[TWIN]**, `image3_text` **[TWIN]**

**prediction:** ["46", "64"]

**answer\_ref:** `46 27 64`

### Representative active cases — mixed_redundant

#### q\_id=4 · Analytical · chart

**Question:** Among the top 10 findings, how many of them are supported by statistical results grouped by the countries?

**gold\_set:** ['image3', 'image3_text', 'image4', 'image4_text', 'image6', 'image6_text', 'image7', 'image7_text']

**retrieved (top-10):** `image6` **[GOLD]**, `text5`, `image7` **[GOLD]**, `image4` **[GOLD]**, `image3` **[GOLD]**, `text11`, `image6_text` **[GOLD TWIN]**, `image5_text` **[TWIN]**, `image1`, `image1_text` **[TWIN]**

**prediction:** Based on the information provided, there are no specific "top 10 findings" mentioned in the given text. However, if we consider the context of the images and the data presented, it appears that the first image provides a breakdown of responses to a survey question across various countries and region…

**answer\_ref:** `4`

#### q\_id=1591 · Application-based · figure|text

**Question:** Which areas do candidates need to focus on for the LinkedIn Recruiter Certification exam, and how does understanding Boolean search results through a Venn diagram relate to these topic areas?

**gold\_set:** ['image7', 'image7_text', 'text4']

**retrieved (top-10):** `image4`, `image4_text` **[TWIN]**, `image3`, `text4` **[GOLD]**, `text10`, `text11`, `text6`, `image3_text` **[TWIN]**, `text5`, `text12`

**prediction:** Candidates need to focus on several key areas for the LinkedIn Recruiter Certification exam, including identifying talent, engaging talent, building a talent pipeline, posting jobs, and maximizing efficiency. Understanding how to construct accurate Boolean search strings, as represented by a Venn di…

**answer\_ref:** `Candidates need to focus on building a talent pipeline, posting jobs, maximizing efficiency, engaging talent, and identifying talent for the LinkedIn Recruiter Certification exam. Understanding Boolean search results, as demonstrated in a Venn diagram, relates to identifying talent using search techniques.`

#### q\_id=75 · Causal · chart|table|text

**Question:** How do voter perceptions of political entities and campaign negativity in the 2016 election relate to each other?

**gold\_set:** ['image1', 'image1_text', 'image4', 'image4_text', 'text12', 'text2']

**retrieved (top-10):** `image3`, `text12` **[GOLD]**, `image1` **[GOLD]**, `text2` **[GOLD]**, `text4`, `image6`, `image4_text` **[GOLD TWIN]**, `text3`, `image1_text` **[GOLD TWIN]**, `text8`

**prediction:** Voter perceptions of political entities and campaign negativity in the 2016 election are closely related. The data shows that a large majority of voters viewed the 2016 campaign as exceptionally negative, with 92% saying there was more mudslinging or negative campaigning than in past elections. This…

**answer\_ref:** `Voter perceptions in the 2016 election were generally negative, with poor grades for political entities and high levels of perceived mudslinging.`

#### q\_id=3 · Comparative · chart|text

**Question:** What percentage of the party that holds the highest total percentage of good policy ideas and high ethical standards reported that they followed the news about congressional elections in their state very closely in the survey of U.S. adults conducted from April 25 to May 1, 2018?

**gold\_set:** ['image4', 'image4_text', 'image7', 'image7_text', 'text2', 'text3', 'text6']

**retrieved (top-10):** `image7` **[GOLD]**, `text2` **[GOLD]**, `image7_text` **[GOLD TWIN]**, `text3` **[GOLD]**, `text5`, `text12`, `text6` **[GOLD]**, `text1`, `text7`, `text8`

**prediction:** Based on the information provided, there is no specific data given about the percentage of the party that holds the highest total percentage of good policy ideas and high ethical standards reporting that they followed the news about congressional elections in their state very closely. The data provi…

**answer\_ref:** `21%`

#### q\_id=0 · Descriptive · chart|text

**Question:** In 2015, how many percentage of surveyed adults, Republicans, and Democrats believed that the goverment was doing very/fairly well in reducing the threat of terrorism? Please write the answer in list format, e.g., ["3","2"]

**gold\_set:** ['image4', 'image4_text', 'image7', 'image7_text', 'text6', 'text9']

**retrieved (top-10):** `image5`, `text9` **[GOLD]**, `text1`, `text6` **[GOLD]**, `text4`, `text5`, `text2`, `image5_text` **[TWIN]**, `image8_text` **[TWIN]**, `image3_text` **[TWIN]**

**prediction:** ["46", "64"]

**answer\_ref:** `46 27 64`

---

## Pipeline B

### Summary

| Variant | n\_active | n\_inactive | mean\_twins (active) | top question\_types |
|---|---|---|---|---|
| gold_redundant | 1548 | 452 | 1.38 | Descriptive=556, Comparative=521, Interpretative=237 |
| negative_redundant | 1433 | 567 | 2.67 | Descriptive=489, Comparative=475, Interpretative=242 |
| mixed_redundant | 1700 | 300 | 3.19 | Descriptive=613, Comparative=562, Interpretative=269 |

### Question-type breakdown (active questions only)

| Question type | gold\_r | neg\_r | mix\_r |
|---|---|---|---|
| Analytical | 177 | 169 | 185 |
| Application-based | 1 | 2 | 2 |
| Causal | 14 | 15 | 15 |
| Comparative | 521 | 475 | 562 |
| Descriptive | 556 | 489 | 613 |
| Inferential | 23 | 28 | 33 |
| Interpretative | 237 | 242 | 269 |
| Procedural | 19 | 13 | 21 |

### Evidence modality breakdown (active questions, top 8)

| Modality | gold\_r | neg\_r | mix\_r |
|---|---|---|---|
| table|text | 524 | 500 | 546 |
| table | 353 | 326 | 361 |
| figure | 202 | 164 | 218 |
| figure|text | 154 | 146 | 197 |
| chart|text | 105 | 104 | 143 |
| chart | 81 | 73 | 89 |
| chart|table|text | 46 | 43 | 50 |
| figure|table|text | 40 | 36 | 41 |

### Representative active cases — gold_redundant

#### q\_id=4 · Analytical · chart

**Question:** Among the top 10 findings, how many of them are supported by statistical results grouped by the countries?

**gold\_set:** ['image3', 'image3_text', 'image4', 'image4_text', 'image6', 'image6_text', 'image7', 'image7_text']

**retrieved (top-10):** `image6_text` **[GOLD TWIN]**, `image7_text` **[GOLD TWIN]**, `image3_text` **[GOLD TWIN]**, `image4_text` **[GOLD TWIN]**, `text5`, `text9`, `image4` **[GOLD]**, `text12`, `text11`, `image6` **[GOLD]**

**prediction:** Five of the top 10 findings are supported by statistical results grouped by countries.

**answer\_ref:** `4`

#### q\_id=1995 · Application-based · table

**Question:** How do the different metering modes and focus modes in Pro Mode enhance photography under various scenarios?

**gold\_set:** ['image4', 'image4_text', 'image5', 'image5_text']

**retrieved (top-10):** `image3`, `text8`, `text3`, `image5_text` **[GOLD TWIN]**, `text9`, `image4_text` **[GOLD TWIN]**, `text6`, `image2`, `text11`, `text1`

**prediction:** The different metering modes and focus modes in Pro Mode enhance photography under various scenarios by allowing users to fine-tune their photos for specific situations. For example, matrix or center-metered mode can be used when shooting natural landscapes with varying light conditions; spot-meters…

**answer\_ref:** `Metering modes (Matrix, Center, Spot) adjust light measurement for landscapes, portraits, and specific areas. Focus modes (AF-S, AF-C, MF) cater to stationary, moving subjects, or manual focus.`

#### q\_id=75 · Causal · chart|table|text

**Question:** How do voter perceptions of political entities and campaign negativity in the 2016 election relate to each other?

**gold\_set:** ['image1', 'image1_text', 'image4', 'image4_text', 'text12', 'text2']

**retrieved (top-10):** `image3`, `image7`, `image1` **[GOLD]**, `image6`, `text10`, `text4`, `text12` **[GOLD]**, `image1_text` **[GOLD TWIN]**, `text2` **[GOLD]**, `image2`

**prediction:** Voter perceptions of political entities and campaign negativity in the 2016 election are related, with a high percentage (94%) saying there was more mudslinging or negative campaigning than past elections. This perception is shared by Clinton voters at an even higher rate compared to Trump supporter…

**answer\_ref:** `Voter perceptions in the 2016 election were generally negative, with poor grades for political entities and high levels of perceived mudslinging.`

#### q\_id=12 · Comparative · table

**Question:** What is the difference in total volume between the rank 1 and rank 19 top albums?

**gold\_set:** ['image2', 'image2_text', 'image3', 'image3_text']

**retrieved (top-10):** `image3_text` **[GOLD TWIN]**, `image2_text` **[GOLD TWIN]**, `image4`, `text2`, `text5`, `text10`, `text6`, `text8`, `text1`, `text11`

**prediction:** The difference in total volume between the rank 1 and rank 19 top albums is:  - Rank: Taylor Swift - "1,608" - Total Volume (in thousands): $523  Therefore,  $|Total\;Volume \ for\:Taylor\ Swifft-\ Total\;Volume \ of\:Rank:\ 19|$ = |(47) |  Hence, Answer: It's $(4.7)$.

**answer\_ref:** `1194000`

#### q\_id=1 · Descriptive · table|text

**Question:** Among the Higher-income seniors, what are the percentage of them go online, has smartphone phone, and own a tablet computer? Please write the answer in the list format and in descend order,e.g., ["9%","8%"] in the Pew Research Center’s Internet Project July 18-September 30, 2013 tracking survey?

**gold\_set:** ['image1', 'image1_text', 'image4', 'image4_text', 'image5', 'image5_text', 'text5', 'text7']

**retrieved (top-10):** `image3`, `image6`, `image7`, `image8`, `image2`, `image5_text` **[GOLD TWIN]**, `image4_text` **[GOLD TWIN]**, `text9`, `text1`, `image1_text` **[GOLD TWIN]**

**prediction:** [90%, 82%, 73%]

**answer\_ref:** `90% 42% 39%`

### Representative active cases — negative_redundant

#### q\_id=4 · Analytical · chart

**Question:** Among the top 10 findings, how many of them are supported by statistical results grouped by the countries?

**gold\_set:** ['image3', 'image4', 'image6', 'image7']

**retrieved (top-10):** `image8_text` **[TWIN]**, `text5`, `text9`, `image4` **[GOLD]**, `text12`, `image5_text` **[TWIN]**, `image1_text` **[TWIN]**, `text11`, `image6` **[GOLD]**, `text2`

**prediction:** Among the top 10 findings, only one of them is supported by statistical results grouped by countries.

**answer\_ref:** `4`

#### q\_id=1591 · Application-based · figure|text

**Question:** Which areas do candidates need to focus on for the LinkedIn Recruiter Certification exam, and how does understanding Boolean search results through a Venn diagram relate to these topic areas?

**gold\_set:** ['image7', 'text4']

**retrieved (top-10):** `image4`, `image3`, `image7` **[GOLD]**, `text10`, `image1`, `image4_text` **[TWIN]**, `text4` **[GOLD]**, `text11`, `text5`, `text6`

**prediction:** Candidates need to focus on the following areas for LinkedIn Recruiter Certification: identifying talent through search, engaging with potential candidates using their presence and InMail features of LinkedIn; building a pipeline by creating an effective Talent Pipeline that includes pipelining tech…

**answer\_ref:** `Candidates need to focus on building a talent pipeline, posting jobs, maximizing efficiency, engaging talent, and identifying talent for the LinkedIn Recruiter Certification exam. Understanding Boolean search results, as demonstrated in a Venn diagram, relates to identifying talent using search techniques.`

#### q\_id=75 · Causal · chart|table|text

**Question:** How do voter perceptions of political entities and campaign negativity in the 2016 election relate to each other?

**gold\_set:** ['image1', 'image4', 'text12', 'text2']

**retrieved (top-10):** `image3`, `image7`, `image1` **[GOLD]**, `image6`, `text10`, `text4`, `text12` **[GOLD]**, `image7_text` **[TWIN]**, `text2` **[GOLD]**, `image2`

**prediction:** Voter perceptions of political entities and campaign negativity in the 2016 election are related.

**answer\_ref:** `Voter perceptions in the 2016 election were generally negative, with poor grades for political entities and high levels of perceived mudslinging.`

#### q\_id=12 · Comparative · table

**Question:** What is the difference in total volume between the rank 1 and rank 19 top albums?

**gold\_set:** ['image2', 'image3']

**retrieved (top-10):** `image6_text` **[TWIN]**, `image4`, `image8_text` **[TWIN]**, `text2`, `image5_text` **[TWIN]**, `text5`, `image4_text` **[TWIN]**, `image1_text` **[TWIN]**, `text10`, `text6`

**prediction:** The difference in total volume between the rank 1 and rank 19 top albums is not provided.

**answer\_ref:** `1194000`

#### q\_id=2 · Descriptive · table

**Question:** How many types of ecosystem players are listed in the slide?

**gold\_set:** ['image1', 'image4']

**retrieved (top-10):** `text3`, `text9`, `image7`, `text12`, `text7`, `image4` **[GOLD]**, `image7_text` **[TWIN]**, `image5_text` **[TWIN]**, `text8`, `text1`

**prediction:** There are 10 types of ecosystem players listed in the slide.

**answer\_ref:** `20`

### Representative active cases — mixed_redundant

#### q\_id=4 · Analytical · chart

**Question:** Among the top 10 findings, how many of them are supported by statistical results grouped by the countries?

**gold\_set:** ['image3', 'image3_text', 'image4', 'image4_text', 'image6', 'image6_text', 'image7', 'image7_text']

**retrieved (top-10):** `image6_text` **[GOLD TWIN]**, `image7_text` **[GOLD TWIN]**, `image3_text` **[GOLD TWIN]**, `image4_text` **[GOLD TWIN]**, `image8_text` **[TWIN]**, `text5`, `text9`, `image4` **[GOLD]**, `text12`, `image5_text` **[TWIN]**

**prediction:** Among the top 10 findings, all of them are supported by statistical results grouped by countries.

**answer\_ref:** `4`

#### q\_id=1591 · Application-based · figure|text

**Question:** Which areas do candidates need to focus on for the LinkedIn Recruiter Certification exam, and how does understanding Boolean search results through a Venn diagram relate to these topic areas?

**gold\_set:** ['image7', 'image7_text', 'text4']

**retrieved (top-10):** `image4`, `image3`, `image7` **[GOLD]**, `text10`, `image1`, `image4_text` **[TWIN]**, `text4` **[GOLD]**, `text11`, `text5`, `text6`

**prediction:** Candidates need to focus on the following areas for LinkedIn Recruiter Certification: identifying talent through search, engaging with potential candidates using their presence and InMail features of LinkedIn; building a pipeline by creating an effective Talent Pipeline that includes pipelining tech…

**answer\_ref:** `Candidates need to focus on building a talent pipeline, posting jobs, maximizing efficiency, engaging talent, and identifying talent for the LinkedIn Recruiter Certification exam. Understanding Boolean search results, as demonstrated in a Venn diagram, relates to identifying talent using search techniques.`

#### q\_id=75 · Causal · chart|table|text

**Question:** How do voter perceptions of political entities and campaign negativity in the 2016 election relate to each other?

**gold\_set:** ['image1', 'image1_text', 'image4', 'image4_text', 'text12', 'text2']

**retrieved (top-10):** `image3`, `image7`, `image1` **[GOLD]**, `image6`, `text10`, `text4`, `text12` **[GOLD]**, `image7_text` **[TWIN]**, `image1_text` **[GOLD TWIN]**, `text2` **[GOLD]**

**prediction:** Voter perceptions of political entities and campaign negativity in the 2016 election are related. The graph shows that a high percentage (92%) said there was more mudslinging or negative campaigning than past elections, which is higher compared to previous years such as Obama's first term when it re…

**answer\_ref:** `Voter perceptions in the 2016 election were generally negative, with poor grades for political entities and high levels of perceived mudslinging.`

#### q\_id=12 · Comparative · table

**Question:** What is the difference in total volume between the rank 1 and rank 19 top albums?

**gold\_set:** ['image2', 'image2_text', 'image3', 'image3_text']

**retrieved (top-10):** `image3_text` **[GOLD TWIN]**, `image2_text` **[GOLD TWIN]**, `image6_text` **[TWIN]**, `image4`, `image8_text` **[TWIN]**, `text2`, `image5_text` **[TWIN]**, `text5`, `image4_text` **[TWIN]**, `image1_text` **[TWIN]**

**prediction:** The difference in total volume between the rank 1 and rank 19 top albums is as follows:  - Rank: The first album has a sales figure of "Total Volume (000)" = [insert value].    For example, if it's Taylor Swift with an unspecified number: Ranking for this artist could be based on their most recent o…

**answer\_ref:** `1194000`

#### q\_id=1 · Descriptive · table|text

**Question:** Among the Higher-income seniors, what are the percentage of them go online, has smartphone phone, and own a tablet computer? Please write the answer in the list format and in descend order,e.g., ["9%","8%"] in the Pew Research Center’s Internet Project July 18-September 30, 2013 tracking survey?

**gold\_set:** ['image1', 'image1_text', 'image4', 'image4_text', 'image5', 'image5_text', 'text5', 'text7']

**retrieved (top-10):** `image3`, `image6`, `image7`, `image8`, `image2`, `image5_text` **[GOLD TWIN]**, `image4_text` **[GOLD TWIN]**, `text9`, `text1`, `image1_text` **[GOLD TWIN]**

**prediction:** [90%, 82%, 73%]

**answer\_ref:** `90% 42% 39%`

---
