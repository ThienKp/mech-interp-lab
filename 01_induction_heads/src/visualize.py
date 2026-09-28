import json
import torch
import matplotlib.pyplot as plt

from model import AttentionModel
from data_synthesis import generate_random_token
from utils import CONTEXT_LENGTH, REPEATED, get_model_filename, get_metrics_filename, ensure_visualizations, get_visualization_filename

DIR_NAME = "mixing_pattern"


def visualize_attention_weights(
        model: AttentionModel,
        same_token_head: tuple,
        induction_head: tuple) -> None:
    """
    Visualize the attention weights of the same-token head and induction head.

    Args:
        model (AttentionModel): The trained attention model.
        same_token_head (tuple): A tuple containing the layer and head indices of the same-token head.
        induction_head (tuple): A tuple containing the layer and head indices of the induction head.
    """
    # Visualize the attention weights
    model.eval()
    with torch.no_grad():
        sequence = generate_random_token(CONTEXT_LENGTH // 4, repeated=REPEATED) * 4 # Generate a random sequence for visualization
        inputs = torch.tensor(sequence[:-1], dtype=torch.long).unsqueeze(0)
        outputs = model(inputs)
        attention_weights = outputs["attention_weights"]  # (1, N_HEADS, seq_length - 1, seq_length - 1) * N_LAYERS
        attention_average = attention_weights[same_token_head[0]][:, same_token_head[1], :, :].mean(dim=0)  # Average over samples
        plt.imshow(attention_average.numpy(), cmap='plasma')
        plt.colorbar()  # Shows the mapping of values to colors
        plt.title(f"Same-token head (Layer {same_token_head[0]} Head {same_token_head[1]})")
        plt.savefig(get_visualization_filename("same_token_head_attention.png"))
        plt.close()  # Close the current figure to avoid overlap

        attention_average = attention_weights[induction_head[0]][:, induction_head[1], :, :].mean(dim=0)  # Average over samples
        plt.imshow(attention_average.numpy(), cmap='plasma')
        plt.colorbar()  # Shows the mapping of values to colors
        plt.title(f"Induction head (Layer {induction_head[0]} Head {induction_head[1]})")
        plt.savefig(get_visualization_filename("induction_head_attention.png"))
        plt.close()  # Close the current figure to avoid overlap

def induction_accuracy_by_head(model: AttentionModel) -> None:
    model.eval()
    with torch.no_grad():
        sequence = []
        pattern_list = []
        pattern_length = CONTEXT_LENGTH // 4
        num_samples = 1000
        layer_heads = [f"L{i}H{j}" for i in range(2) for j in range(4)]
        induction_position_accuracy = {f"pattern_{i + 1}": [] for i in range(4)}

        for i in range(num_samples):
            pattern = generate_random_token(pattern_length, repeated=REPEATED)
            pattern_list.append(pattern)
            sequence += pattern * 4
        inputs = torch.tensor(sequence, dtype=torch.long).view(num_samples, CONTEXT_LENGTH) # (num_samples, seq_length)
        inputs = inputs[:, :-1]

        for layer in range(2):
            for head in range(4):
                outputs = model(inputs, [(layer, head)])
                logits = outputs["logits"] # (num_samples, seq_length, VOCAB_SIZE)
                total_predictions = (pattern_length - 1) * num_samples

                # Induction-position accuracy and loss
                for i in range(4):
                    pred = logits[:, i * pattern_length:(i + 1) * pattern_length - 1, :]
                    induction_position_accuracy[f"pattern_{i + 1}"].append((pred.argmax(dim=-1) == torch.tensor(pattern_list)[:, 1:]).sum().item())
                    induction_position_accuracy[f"pattern_{i + 1}"][-1] /= total_predictions / 100
                    induction_position_accuracy[f"pattern_{i + 1}"][-1] = round(induction_position_accuracy[f"pattern_{i + 1}"][-1], 2)

        fig, ax = plt.subplots(layout='constrained')
        ax.grouped_bar(induction_position_accuracy, tick_labels=layer_heads, group_spacing=1)

        # Add some text for labels, title, etc.
        ax.set_ylabel('Induction Accuracy (%)')
        ax.set_title('Induction Accuracy by Head')
        ax.legend(loc='upper left', ncols=4)
        ax.set_ylim(0, 115)
        plt.savefig(get_visualization_filename("induction_accuracy_by_head.png"))
        plt.close()  # Close the current figure to avoid overlap

if __name__ == "__main__":
    model = AttentionModel()
    ensure_visualizations()
    # Load the trained model weights here if available
    model.load_state_dict(torch.load(get_model_filename(DIR_NAME)))
    with open(get_metrics_filename(DIR_NAME)) as f:
        metrics = json.load(f)
        same_token_head = metrics["ablation_studies"]["same_token_head"]["layer_head"]
        induction_head = metrics["ablation_studies"]["induction_head"]["layer_head"]
        visualize_attention_weights(model, same_token_head, induction_head)
        induction_accuracy_by_head(model)