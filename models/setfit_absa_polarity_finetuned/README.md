---
tags:
- setfit
- absa
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: recently purchased the mi power bank 3i 20000mah and:charged my samsung m53
    twice once full charged.,i recently purchased the mi power bank 3i 20000mah and
    have been extremely impressed with its performance.
- text: hdmi cable for boat soundbar and lg smart:perfect hdmi cable for boat soundbar
    and lg smart tv,this product is overpriced,value for money good quality product,quality
    product,good ,good quality,good,it's ok to purchase for and as arc port
- text: like it's designe speed and build quality:when we plug in mobile and then
    in pc i am getting some kind of error but files are intact and able to copy it,i
    like it's designe speed and build quality but there is a negetive point after
    you flip side for 10 to 20 time you will find that tha flip cover is starting
    to get loose,so far so good, it's early to right a review, as per my usage i will
    update accordingly,good,user friendly
- text: really poor in quality..a rusty:entire product including the beating machine
    nd other pair of dough hooker is absolutely perfect but the important pair of
    steel beater is really poor in quality..a rusty pair of beater 've been delivered
    nd that's too very disappointing...
- text: weight and average quality.:recommend for buy,average product light weight
    and average quality.
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
| Label    | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|:---------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| negative | <ul><li>'wall clocks,charger is not good:40 290 ,charging does not last for more than 3 days with out use after full recharge,battery does not last even one week even in wall clocks,charger is not good because no idicate light after battery fully charge,get higher wattage.....'</li><li>'full recharge,battery does not last:40 290 ,charging does not last for more than 3 days with out use after full recharge,battery does not last even one week even in wall clocks,charger is not good because no idicate light after battery fully charge,get higher wattage.....'</li><li>"not true, button press sound is there.overall it:i am using it from 1 month, it works decent.size is bit small, so it does not provide a good ergonomic.as they say clicks are noiseless, that is not true, button press sound is there.overall it's considerable and worth buying it.,like,light weight and work well,good,hands down, logitech is the best in mouses."</li></ul>                                                                                        |
| neutral  | <ul><li>'use after full recharge,battery does:40 290 ,charging does not last for more than 3 days with out use after full recharge,battery does not last even one week even in wall clocks,charger is not good because no idicate light after battery fully charge,get higher wattage.....'</li><li>'battery drain revisit few:battery drain revisit few of the settings on your device, like animation, transition effect etc to get most out of your battery.'</li><li>"have to consider cost while writing review:using a short cable while charging puts strain on the cable and phone's charging port, which is damaging for both, and also decreases life of both the cable and charging port.....they say you have to consider cost while writing review..."</li></ul>                                                                                                                                                                                                                                                                                        |
| positive | <ul><li>"like it's designe speed and build quality:when we plug in mobile and then in pc i am getting some kind of error but files are intact and able to copy it,i like it's designe speed and build quality but there is a negetive point after you flip side for 10 to 20 time you will find that tha flip cover is starting to get loose,so far so good, it's early to right a review, as per my usage i will update accordingly,good,user friendly"</li><li>"speed and build quality but there is:when we plug in mobile and then in pc i am getting some kind of error but files are intact and able to copy it,i like it's designe speed and build quality but there is a negetive point after you flip side for 10 to 20 time you will find that tha flip cover is starting to get loose,so far so good, it's early to right a review, as per my usage i will update accordingly,good,user friendly"</li><li>'perfect with no lag.:no more changing of cables every year or soo.,as i need for apple carplay support this is perfect with no lag.'</li></ul> |

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
| Word count   | 3   | 47.1404 | 222 |

| Label    | Training Sample Count |
|:---------|:----------------------|
| negative | 52                    |
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
| 0.0116 | 1    | 0.1795        | -               |
| 0.5814 | 50   | 0.137         | -               |

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