---
tags:
- setfit
- absa
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: slow charging, charging speed:slow charging, charging speed is not fast,not
    happy with this product bcoz this is a highly priced as compared to its quality
    and the wire is not actually retracted easily.
- text: so effective price of popio cable:so effective price of popio cable is rs.500
    for 1.5 meter cable...popio cable is 1 meters long against 1.5 meters of original
    cable...
- text: say clicks are noiseless, that is:i am using it from 1 month, it works decent.size
    is bit small, so it does not provide a good ergonomic.as they say clicks are noiseless,
    that is not true, button press sound is there.overall it's considerable and worth
    buying it.,like,light weight and work well,good,hands down, logitech is the best
    in mouses.
- text: but price is slightly high:but price is slightly high for this product..,one
    got damaged immediately,very good product if screen little bit more brighter,
    over all nice product !,,this product is very good, its quality is also very good
    and it proved to be very useful for children, purchased a very good product,brightness
    is good, writing is smooth.easy to use.has lock button at behind to restrict unwanted
    erase.butwhile writing dots apper near by drawn text and lines automatically without
    touching there, so may b not suitable for precise calculation and maths like stuff,
    example u writing 42 and it may appear 4.2 as dot automatically appear between
    or near by them at certain places.dots problem more near border and corner.same
    product same package can be seen in much lower price if you google properly.my
    use was just for rough work soi didn't request for replace or return as seller
    send from so far via delivery and i feel it may be uneconomical for seller to
    receive and return via delivery service.
- text: justifies it's value, product for:i would have really appreciate if the quality
    of beaters can be improved coz the entire product other than this justifies it's
    value, product for making cake cream....i have used this product more than 8 months.....no
    issue...no complaint.....good working,it's good if used properly.,i like this
    product.,its such a good product by kent.
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: false
model-index:
- name: SetFit Polarity Model
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: Unknown
      type: unknown
      split: test
    metrics:
    - type: accuracy
      value: 0.8636363636363636
      name: Accuracy
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
| Label    | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| positive | <ul><li>'green black tea.totally value for money.:one for coffee another for green black tea.totally value for money.'</li><li>'good iron, performance, look and:nice,good iron, performance, look and shape is very good,i like this product,yes,working well now.,nice product,acch hai,good'</li><li>"like it's designe speed and build quality:when we plug in mobile and then in pc i am getting some kind of error but files are intact and able to copy it,i like it's designe speed and build quality but there is a negetive point after you flip side for 10 to 20 time you will find that tha flip cover is starting to get loose,so far so good, it's early to right a review, as per my usage i will update accordingly,good,user friendly"</li></ul>                                                                                                                         |
| negative | <ul><li>'and quickly loses power.:does not have enough oomph and quickly loses power.'</li><li>",value for money good quality product:perfect hdmi cable for boat soundbar and lg smart tv,this product is overpriced,value for money good quality product,quality product,good ,good quality,good,it's ok to purchase for and as arc port"</li><li>'parameters affect the performance and the difference:generally, it is worth going for the higher memory and higher ram versions as these parameters affect the performance and the difference in cost is relatively small.'</li></ul>                                                                                                                                                                                                                                                                                                 |
| neutral  | <ul><li>"2 years, cost of which at:it puts strain on both the points, the charger side and mobile side, which again makes it vulnerable to cracks...this raises a question on longevity of the cable.....let's do some calculations.....my original oneplus cable lasted for 2 years, cost of which at present is rs.1100...."</li><li>', which means price may rise ...:the rs.1100 cable online is 1.5 meters long, which makes it easy to work on phone while it is plugged.....popio cable costs rs.400 at present but they show it 999 , which means price may rise ...'</li><li>"have to consider cost while writing review:using a short cable while charging puts strain on the cable and phone's charging port, which is damaging for both, and also decreases life of both the cable and charging port.....they say you have to consider cost while writing review..."</li></ul> |

## Evaluation

### Metrics
| Label   | Accuracy |
|:--------|:---------|
| **all** | 0.8636   |

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
| Word count   | 3   | 52.3509 | 205 |

| Label    | Training Sample Count |
|:---------|:----------------------|
| negative | 8                     |
| neutral  | 10                    |
| positive | 39                    |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 2
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
| 0.0667 | 1    | 0.104         | -               |
| 1.0    | 15   | -             | 0.2075          |

### Framework Versions
- Python: 3.11.9
- SetFit: 1.1.3
- Sentence Transformers: 5.5.0
- spaCy: 3.8.11
- Transformers: 4.57.6
- PyTorch: 2.12.0+cpu
- Datasets: 4.8.5
- Tokenizers: 0.22.2

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