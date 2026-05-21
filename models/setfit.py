from setfit import AbsaModel, AbsaTrainer, TrainingArguments
from datasets import load_dataset
from transformers import EarlyStoppingCallback

# You can initialize a AbsaModel using one or two SentenceTransformer models, or two ABSA models
model = AbsaModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# The training/eval dataset must have `text`, `span`, `polarity`, and `ordinal` columns
dataset = load_dataset("tomaarsen/setfit-absa-semeval-laptops")
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

args = TrainingArguments(
    output_dir="models",
    use_amp=True,
    batch_size=256,
    eval_steps=50,
    save_steps=50,
    load_best_model_at_end=True,
)

trainer = AbsaTrainer(
    model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)
trainer.train()

metrics = trainer.evaluate(eval_dataset)
print(metrics)

trainer.push_to_hub("tomaarsen/setfit-absa-laptops")

from setfit import AbsaModel

# Download from Hub and run inference
model = AbsaModel.from_pretrained(
    "tomaarsen/setfit-absa-laptops-aspect",
    "tomaarsen/setfit-absa-laptops-polarity",
)

# Run inference
preds = model([
    "Boots up fast and runs great!",
    "The screen shows great colors.",
])