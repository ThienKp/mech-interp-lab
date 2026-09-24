import json
import torch

from model import AttentionModel
from data_synthesis import generate_sequence_parts, generate_sequence
from utils import CONTEXT_LENGTH, VOCAB_SIZE, REPEATED, N_HEADS, N_LAYERS, get_model_filename, get_metrics_filename

# Directory name for retrieving the model
DIR_NAME: str = "MHA_without_replacement"

def acc_and_loss(logits, targets, num_samples, second_pattern_index, pattern_list) -> tuple:
    """
    Calculate accuracy and loss for next-token prediction.

    Args:
        logits (torch.Tensor): Logits output from the model of shape (batch_size, seq_length, vocab_size).
        targets (torch.Tensor): Ground truth targets of shape (batch_size, seq_length).
        num_samples (int): Number of samples in the batch.
        second_pattern_index (list): List of indices indicating the start of the second pattern in each sample.

    Returns:
        tuple: Overall accuracy, overall loss, induction-position accuracy, induction-position loss.
    """
    # Next-token accuracy
    predictions = torch.argmax(logits, dim=-1)  # (num_samples, seq_length - 1)
    correct_predictions = (predictions == targets).sum().item()
    total_predictions = targets.numel()
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Model Next-token Accuracy: {overall_accuracy * 100:.2f}%")

    # Next-token loss
    criterion = torch.nn.CrossEntropyLoss()
    overall_loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)).item()
    print(f"Model Next-token Loss: {overall_loss:.4f}")

    # Induction-position accuracy and loss
    correct_predictions = 0
    total_predictions = 0
    loss = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    for i in range(num_samples):
        second_start = second_pattern_index[i]
        pattern = pattern_list[i]
        correct_predictions += (logits[i, second_start:second_start + len(pattern) - 1].argmax(dim=-1) == torch.tensor(pattern[1:])).sum().item()
        loss += criterion(logits[i, second_start:second_start + len(pattern) - 1], torch.tensor(pattern[1:])).item()
        total_predictions += len(pattern) - 1
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    loss /= total_predictions if total_predictions > 0 else 1
    print(f"Induction-position accuracy: {accuracy * 100:.2f}%")
    print(f"Induction-position loss: {loss:.4f}")

    return overall_accuracy, overall_loss, accuracy, loss

def analyze_model(model, num_samples: int = 1000, seq_length: int = CONTEXT_LENGTH) -> None:
    """
    Analyze the performance of the model on next-token prediction and induction positions.

    Args:
        model (AttentionModel): The trained attention model to analyze.
        num_samples (int): Number of samples to generate for analysis.
        seq_length (int): Length of each generated sequence.
    """
    model.eval()
    metrics = {}

    with torch.no_grad():
        sequence = []
        first_pattern_index = []
        second_pattern_index = []
        pattern_list = []

        for i in range(num_samples):
            prefix, pattern, gap, suffix = generate_sequence_parts(seq_length, repeated=REPEATED)
            first_pattern_index.append(len(prefix))
            second_pattern_index.append(len(prefix) + len(pattern) + len(gap))
            pattern_list.append(pattern)
            sequence += prefix + pattern + gap + pattern + suffix
        inputs = torch.tensor(sequence, dtype=torch.long).view(num_samples, seq_length) # (num_samples, seq_length)
        targets = inputs[:, 1:] # Shifted target for next token prediction
        inputs = inputs[:, :-1]

        outputs = model(inputs)
        logits = outputs["logits"]  # (num_samples, seq_length - 1, vocab_size)

        print("Experiment 1: Accuracy and loss of the model in predicting the next token and induction positions.")
        print("------------------------------------------------------------")
        next_token_accuracy, next_token_loss, induction_position_accuracy, induction_position_loss = acc_and_loss(logits, targets, num_samples, second_pattern_index, pattern_list)
        metrics["next_token_accuracy"] = next_token_accuracy
        metrics["next_token_loss"] = next_token_loss
        metrics["induction_position_accuracy"] = induction_position_accuracy
        metrics["induction_position_loss"] = induction_position_loss

        # Indicate which positions contribute to attentions head the most
        print("\nExperiment 2: Attention weights analysis for each layer and head.")
        print("------------------------------------------------------------")
        attention_weights = outputs["attention_weights"] # (num_samples, N_HEADS, seq_length - 1, seq_length - 1) * N_LAYERS
        copying_head = {"layer": None, "head": None, "attention_average": 0}
        induction_head = {"layer": None, "head": None, "attention_average": 0}
        
        for layer in range(N_LAYERS):
            metrics[f"layer_{layer}_attention"] = {}
            for head in range(N_HEADS):
                same_token_attention = 0
                next_token_attention = 0
                total_pattern = 0
                for i in range(num_samples):
                    first_start = first_pattern_index[i]
                    second_start = second_pattern_index[i]
                    pattern = pattern_list[i]
                    for j in range(len(pattern) - 1):
                        same_token_attention += attention_weights[layer][i, head, second_start + j, first_start + j].item()
                        next_token_attention += attention_weights[layer][i, head, second_start + j, first_start + j + 1].item()
                    total_pattern += len(pattern) - 1
                same_token_attention /= total_pattern
                next_token_attention /= total_pattern

                if same_token_attention > copying_head["attention_average"]:
                    copying_head["layer"] = layer
                    copying_head["head"] = head
                    copying_head["attention_average"] = same_token_attention
                if next_token_attention > induction_head["attention_average"]:
                    induction_head["layer"] = layer
                    induction_head["head"] = head
                    induction_head["attention_average"] = next_token_attention

                print(f"Layer {layer}, Head {head}: Same token attention: {same_token_attention:.4f}, Next token attention: {next_token_attention:.4f}")
                metrics[f"layer_{layer}_attention"][f"head_{head}"] = {
                    "same_token_attention": same_token_attention,
                    "next_token_attention": next_token_attention
                }
        print("\nHighlights:")
        print(f"Copying head: Layer {copying_head['layer']}, Head {copying_head['head']}, Attention average: {copying_head['attention_average']:.4f}")
        print(f"Induction head: Layer {induction_head['layer']}, Head {induction_head['head']}, Attention average: {induction_head['attention_average']:.4f}")

        # Cut the model at the copying head and induction head and analyze the performance
        print("\nExperiment 3: Analyzing the performance of the model when cutting at the copying head or induction head.")
        print("------------------------------------------------------------")
        metrics["ablation_studies"] = {}
        metrics["ablation_studies"]["copying_head"] = {}
        print(f"Cutting at copying head: Layer {copying_head['layer']}, Head {copying_head['head']}")
        outputs = model(inputs, ablate_heads=[(copying_head["layer"], copying_head["head"])])
        logits = outputs["logits"]

        next_token_accuracy, next_token_loss, induction_position_accuracy, induction_position_loss = acc_and_loss(logits, targets, num_samples, second_pattern_index, pattern_list)
        metrics["ablation_studies"]["copying_head"]["next_token_accuracy"] = next_token_accuracy
        metrics["ablation_studies"]["copying_head"]["next_token_loss"] = next_token_loss
        metrics["ablation_studies"]["copying_head"]["induction_position_accuracy"] = induction_position_accuracy
        metrics["ablation_studies"]["copying_head"]["induction_position_loss"] = induction_position_loss

        metrics["ablation_studies"]["induction_head"] = {}
        print(f"\nCutting at induction head: Layer {induction_head['layer']}, Head {induction_head['head']}")
        outputs = model(inputs, ablate_heads=[(induction_head["layer"], induction_head["head"])])
        logits = outputs["logits"]

        next_token_accuracy, next_token_loss, induction_position_accuracy, induction_position_loss = acc_and_loss(logits, targets, num_samples, second_pattern_index, pattern_list)
        metrics["ablation_studies"]["induction_head"]["next_token_accuracy"] = next_token_accuracy
        metrics["ablation_studies"]["induction_head"]["next_token_loss"] = next_token_loss
        metrics["ablation_studies"]["induction_head"]["induction_position_accuracy"] = induction_position_accuracy
        metrics["ablation_studies"]["induction_head"]["induction_position_loss"] = induction_position_loss

    with open(get_metrics_filename(DIR_NAME), 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    model = AttentionModel()
    # Load the trained model weights here if available
    model.load_state_dict(torch.load(get_model_filename(DIR_NAME)))
    print()
    analyze_model(model)
    print()