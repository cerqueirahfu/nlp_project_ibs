---
tags:
- setfit
- absa
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: none of the flagship grade features are skipped here:a word of advice if you
    are planning to buy a new phone, it would be helpful if you list out your requirements
    so that you do not end up paying for features that you do not intend to use.,with
    5g connectivity, a 120hz display and solid performance, the galaxy m33 5g is samsung
    s best value smartphone yet under 20k segmentpros almost half the price of galaxy
    a53exynos 1280 is a solid performerexcellent 120hz display with smooth experience
    across the boardhuge performance uplift over m32one ui 4.1 based on android 12
    has tons of good features to love, none of the flagship grade features are skipped
    here apart from dex mode and spen features , zero lag or stuttering in 120 hz
    mode with ease of use and seemless switching of apps without any crash or reloading
    of apps, makes using one ui pleasure to surf acrossyou get features like a screen
    recorder, video call effects, game launcher, link to windows, dual messenger,
    quick share, music share, and secure folder, along with many others.
- text: good but low audio volume.it's better:stylish design and finishing also good
    but low audio volume.it's better to use extra speakers...,good quality for the
    price,it was amazing to see it in big picture at home..
- text: of handsets without chargers because old chargers:i am willing to live with
    these perceived shortcomings, so long as the m33 meets my requirements.i was glad
    to find that samsung has taken an environmentally friendly step of offering many
    models of handsets without chargers because old chargers, like old handsets, add
    to e waste.
- text: july 2019, dash charge stopped working on:charging rapidly thankful,2nd purchase
    review....purchased on 17 july 2019, dash charge stopped working on 15 september
    2019...
- text: the cable stopped working in 7 months:the cable stopped working in 7 months
    contacted customer support and they ensured a replacement soon.
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: false
---

# SetFit Polarity Model

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Aspect Based Sentiment Analysis (ABSA). A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification. In particular, this model is in charge of classifying aspect polarities.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

This model was trained within the context of a larger system for ABSA, which looks like so:

1. Use a spaCy model to select possible aspect span candidates.
2. Use a SetFit model to filter these possible aspect span candidates.
3. **Use this SetFit model to classify the filtered aspect span candidates.**

## Model Details

### Model Description
- **Model Type:** SetFit
<!-- - **Sentence Transformer:** [Unknown](https://huggingface.co/unknown) -->
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **spaCy Model:** en_core_web_sm
- **SetFitABSA Aspect Model:** [setfit-absa-aspect](https://huggingface.co/setfit-absa-aspect)
- **SetFitABSA Polarity Model:** [models\setfit_absa_polarity_finetuned](https://huggingface.co/models\setfit_absa_polarity_finetuned)
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 3 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label    | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| negative | <ul><li>'won t feel cheap at all.:the screen has big bezels, however, the good part is that the screen goes completely back when it is off, so you won t notice the bezels, and the screen quality won t feel cheap at all.'</li><li>'is fairly even sound, but not:it is fairly even sound, but not a significant enough improvement in bass over the 200.'</li><li>'good product but cost is more.:but packaging is not good feeling like seller gave is used cable.,good product,good product but cost is more.,original cable,i bought this cable at 129.'</li></ul> |
| positive | <ul><li>'....since the price is now reduced:it used to charge in one position only, and charging used to stop with little movement...finally had to replace it with new one after 6 7 months....since the price is now reduced to 300, so i will say good product in this price...'</li><li>',good,value for money.:looks fine waiting for durable report,good,value for money.'</li><li>',value for money.:looks fine waiting for durable report,good,value for money.'</li></ul>                                                                                       |
| neutral  | <ul><li>'buy a rechargeable battery and charger.:so those who are buying this product should also buy a rechargeable battery and charger.'</li><li>'rechargeable battery and charger.:so those who are buying this product should also buy a rechargeable battery and charger.'</li><li>'battery drain revisit few:battery drain revisit few of the settings on your device, like animation, transition effect etc to get most out of your battery.'</li></ul>                                                                                                           |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import AbsaModel

# Download from the 🤗 Hub
model = AbsaModel.from_pretrained(
    "setfit-absa-aspect",
    "models\setfit_absa_polarity_finetuned",
)
# Run inference
preds = model("The food was great, but the venue is just way too busy.")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median  | Max |
|:-------------|:----|:--------|:----|
| Word count   | 3   | 42.0619 | 222 |

| Label    | Training Sample Count |
|:---------|:----------------------|
| negative | 131                   |
| neutral  | 50                    |
| positive | 126                   |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 3
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0086 | 1    | 0.1024        | -               |
| 0.4310 | 50   | 0.1806        | -               |
| 0.8621 | 100  | 0.1276        | -               |

### Framework Versions
- Python: 3.11.9
- SetFit: 1.1.3
- Sentence Transformers: 3.1.1
- spaCy: 3.7.5
- Transformers: 4.45.2
- PyTorch: 2.12.0+cpu
- Datasets: 5.0.0
- Tokenizers: 0.20.3

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->