# Qualitative Examples — MMDocRAG Full Eval (original variant)

Source: `results/full_eval/{a,b}/` — original variant, n=2000.
Scores shown as F1 (answer token overlap, 0–1).

---

## Category 1 — Pipeline A verbose-correct, Pipeline B concise-wrong

### Example 1 &nbsp; (q\_id=30)

**Reference answer:** `Latinos interviewed by cellphone`

**Pipeline A** (F1=0.167)

> According to the report, the Latinos interviewed by cellphone are greater in population compared to the foreign-born Latinos. The report mentions that the sample consisted of a landline sampling frame (yielding 449 completed interviews) and a cellphone sampling frame (1,051 interviews). Therefore, the Latinos interviewed by cellphone represent a larger portion of the sample.

**Pipeline B** (F1=0.286)

> Foreign born Latinos

### Example 2 &nbsp; (q\_id=985)

**Reference answer:** `63`

**Pipeline A** (F1=0.118)

> The median exit valuation in the USA was $63 million more than in Europe at the time of the presentation.

**Pipeline B** (F1=0.000)

> $107M

### Example 3 &nbsp; (q\_id=1228)

**Reference answer:** `post-hoc correction`

**Pipeline A** (F1=0.071)

> The survey provides a detailed review of various correction methodologies, categorizing them into training-time correction, generation-time correction, and post-hoc correction. While the survey mentions the existence of representative papers for each category, it does not explicitly state which category has the most representative papers. Therefore, based on the information provided, it cannot be determined which category has the most representative papers.

**Pipeline B** (F1=0.182)

> Generation-time correction has the most representative papers in this survey.

### Example 4 &nbsp; (q\_id=1561)

**Reference answer:** `4`

**Pipeline A** (F1=0.000)

> Logic-LM (without self-refinement) outperforms the two baseline models in all five datasets when using GPT-4 as the base language model.

**Pipeline B** (F1=0.000)

> Four

### Example 5 &nbsp; (q\_id=1733)

**Reference answer:** `0-375 miles`

**Pipeline A** (F1=0.111)

> The red color in the chart represents an approximate distance from the Mississippi River ranging from 0 to 375 miles.

**Pipeline B** (F1=0.400)

> 0 - 375 miles

---

## Category 2 — Pipeline B concise wrong number

### Example 1 &nbsp; (q\_id=3)

**Reference answer:** `21%`

**Pipeline A** (F1=0.000)

> The Republican Party holds the highest total percentage of good policy ideas and high ethical standards, with 50% for each. According to the survey data, 19% of Americans said they were following news about the candidates and campaign in their state or district very closely. Since the Republican Party is the one with the highest percentages for both good policy ideas and high ethical standards, we can conclude that 19% of the Republican Party's supporters reported following the news about congressional elections in their state very closely.

**Pipeline B** (F1=0.000)

> 50

### Example 2 &nbsp; (q\_id=887)

**Reference answer:** `(4, 233, 483, 160)`

**Pipeline A** (F1=0.000)

> The total number of outstanding shares of Exxon Mobil Corporation as of January 31, 2021, is 4,233,483,160.

**Pipeline B** (F1=0.000)

> 4,233.

### Example 3 &nbsp; (q\_id=1112)

**Reference answer:** `10%`

**Pipeline A** (F1=0.000)

> According to the information provided, 36% of Republicans are not at all confident that Trump keeps his own business interests separate from the decisions he makes as president.

**Pipeline B** (F1=0.000)

> 21

### Example 4 &nbsp; (q\_id=1299)

**Reference answer:** `71.02 75.38`

**Pipeline A** (F1=0.000)

> ["1", "2", "3", "4", "5"]

**Pipeline B** (F1=0.000)

> [0.15, 0.34697289e-05]

### Example 5 &nbsp; (q\_id=1561)

**Reference answer:** `4`

**Pipeline A** (F1=0.000)

> Logic-LM (without self-refinement) outperforms the two baseline models in all five datasets when using GPT-4 as the base language model.

**Pipeline B** (F1=0.000)

> Four

---

## Category 3 — Both pipelines fail

### Example 1 &nbsp; (q\_id=2)

**Reference answer:** `20`

**Pipeline A** (F1=0.000)

> There are seven types of ecosystem players listed in the slide.

**Pipeline B** (F1=0.000)

> There are 10 types of ecosystem players listed in the slide.

### Example 2 &nbsp; (q\_id=858)

**Reference answer:** `24.26`

**Pipeline A** (F1=0.000)

> The information provided does not include the FY2019 fixed asset turnover ratio for Activision Blizzard. The text discusses performance share units, stock options, and MAUs but does not provide financial ratios like the fixed asset turnover ratio. To answer this question, additional financial statements or reports from Activision Blizzard would be required.

**Pipeline B** (F1=0.000)

> The FY2019 fixed asset turnover ratio for Activision Blizzard is 3.45, calculated by dividing the revenue of $678 million in fiscal year (FY) ending December 31st ,by its average total assets over that period which was approximately ($(US$).

### Example 3 &nbsp; (q\_id=1201)

**Reference answer:** `UC Browser Tiếng Việt`

**Pipeline A** (F1=0.000)

> The app "Đồ sát mobile" is in the top 10 Vietnamese Android apps on the APPOTA platform but is not in the top 10 Vietnamese iOS apps.

**Pipeline B** (F1=0.000)

> Zing MP3

### Example 4 &nbsp; (q\_id=1534)

**Reference answer:** `Including venue information improves CLS performance, while removing abstract reduces overall scores.`

**Pipeline A** (F1=0.057)

> The inclusion or exclusion of specific features impacts the performance of the SPECTER model differently across various tasks. For instance, removing the abstract from the textual input and relying solely on the title results in a significant decrease in performance. Adding authors as an input alongside the title and abstract also hurts performance, possibly due to the sparsity of author names in the corpus, making it difficult for the model to infer document-level relatedness. Similarly, adding venues slightly decreases performance, except on document classification tasks. These findings suggest that the choice of features is crucial and should be carefully considered depending on the specific task at hand.

**Pipeline B** (F1=0.127)

> The inclusion of specific features, such as abstracts and authors in the input to SPECTER model can impact its performance. Removing metadata fields like titles or adding venues slightly decreases overall average per-formance across all metrics on different tasks except for document classification task which makes sense because we would expect venue information not be semantically related with each other leading them into noisy correlation when tokenized using Wordpieces

### Example 5 &nbsp; (q\_id=1788)

**Reference answer:** `Yes.`

**Pipeline A** (F1=0.000)

> The given text does not provide specific information about the majority vote requirements for approving international treaties in Costa Rica. Therefore, based solely on the information provided, it cannot be determined if Costa Rica requires a two-thirds majority of all members for approving international treaties.

**Pipeline B** (F1=0.143)

> Yes, Costa Rica requires a two-thirds majority of all members for approving international treaties.

---

## Category 4 — Both pipelines succeed

### Example 1 &nbsp; (q\_id=295)

**Reference answer:** `No`

**Pipeline A** (F1=1.000)

> No

**Pipeline B** (F1=1.000)

> No.

### Example 2 &nbsp; (q\_id=828)

**Reference answer:** `The total stockholders' equity increased from $5,003 million in 2016 to $33,885 million in 2020.`

**Pipeline A** (F1=0.929)

> The total stockholders' equity increased from $15,605 million in 2016 to $33,885 million in 2020.

**Pipeline B** (F1=0.929)

> The total stockholders' equity increased from $15,605 million in 2016 to $33,885 million in 2020.

### Example 3 &nbsp; (q\_id=1286)

**Reference answer:** `The Joint goal accuracy in the "Train" domain improved by 9.87% when using the GEM fine-tuning strategy compared to training from scratch (54.31% - 44.24%).`

**Pipeline A** (F1=0.900)

> The Joint goal accuracy in the "Train" domain improved by 15.59% when using the GEM fine-tuning strategy compared to training from scratch.

**Pipeline B** (F1=0.837)

> The Joint goal accuracy in the "Train" domain improved by $15.6\%$ when using GEM fine-tuning strategy compared to training from scratch ($49.38 - 27 = \boxed{0}\%).

### Example 4 &nbsp; (q\_id=1387)

**Reference answer:** `The two sub-modules within the news encoder are a title encoder and a topic encoder.`

**Pipeline A** (F1=0.815)

> The two sub-modules within the news encoder, as depicted in Figure 2, are the title encoder and the topic encoder.

**Pipeline B** (F1=1.000)

> The two sub-modules within the news encoder are a title encoder and topic encoder.

### Example 5 &nbsp; (q\_id=1502)

**Reference answer:** `In Figure 2(a), there is a clear clustering according to the supersense part-of-speech.`

**Pipeline A** (F1=0.909)

> In Figure 2(a), a clear clustering according to the supersense part-of-speech is observed.

**Pipeline B** (F1=0.909)

> A clear clustering according to the supersense part-of-speech is apparent in Figure 2(a).
