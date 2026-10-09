
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()

tests = [
    {
        "claim": "The Sun revolves around the Earth.",
        "evidence": (
            "Earth revolves around the Sun due to the "
            "gravitational attraction of the Sun."
        ),
    },
    {
        "claim": "The Sun revolves around the Earth.",
        "evidence": (
            "The Earth orbits the Sun once approximately "
            "every 365.25 days."
        ),
    },
    {
        "claim": "The Sun revolves around the Earth.",
        "evidence": (
            "The Sun revolves around the Earth."
        ),
    },
]

id2label = model.config.id2label
print("Label mapping:", id2label)

for test in tests:
    # NLI convention: premise first, hypothesis second.
    inputs = tokenizer(
        test["evidence"],
        test["claim"],
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probs).item())

    print("\nClaim:", test["claim"])
    print("Evidence:", test["evidence"])
    print("Prediction:", id2label[predicted_id])
    print("Probabilities:")

    for i, label in id2label.items():
        print(f"  {label}: {probs[i].item():.4f}")
