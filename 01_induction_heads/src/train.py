import json
import random
import torch
from tqdm import trange

from model import AttentionModel
from data_synthesis import generate_sequence, generate_random_token, generate_multiple_patterns
from utils import SEED, CONTEXT_LENGTH, VOCAB_SIZE, VOCAB_LIST, D_MODEL, N_HEADS, N_LAYERS, REPEATED, ensure_model_database, generate_new_save, get_config_filename, get_model_filename

random.seed(SEED)

def train_model(
        model,
        optimizer,
        criterion,
        num_epochs: int = 100000,
        batch_size: int = 64,
        seq_length: int = CONTEXT_LENGTH,
        pattern_chance: float = 0.9,
        miscellaneous_chance: bool = False,
        to_save: bool = True
    ) -> None:
    model.train()
    for _ in trange(num_epochs, desc="Training Epochs"):\
        # Generate a random sequence of tokens
        sequence = []
        for _ in range(batch_size):
            if miscellaneous_chance:
                number = random.random()
                if number < pattern_chance / 3:
                    sequence += generate_multiple_patterns(seq_length, num_patterns=2, repeated=REPEATED)
                elif number < pattern_chance / 3 * 2:
                    sequence += generate_multiple_patterns(seq_length, num_patterns=3, repeated=REPEATED)
                elif number < pattern_chance:
                    sequence += generate_multiple_patterns(seq_length, num_patterns=4, repeated=REPEATED)
                else:
                    sequence += generate_random_token(seq_length, repeated=REPEATED)
            else:
                if random.random() < pattern_chance:
                    sequence += generate_sequence(seq_length, repeated=REPEATED)
                else:
                    sequence += generate_random_token(seq_length, repeated=REPEATED)
        inputs = torch.tensor(sequence, dtype=torch.long).view(batch_size, seq_length) # (batch_size, seq_length)
        targets = inputs[:, 1:] # Shifted target for next token prediction
        inputs = inputs[:, :-1]

        # Forward pass
        outputs = model(inputs)
        logits = outputs["logits"] # (batch_size, seq_length - 1, VOCAB_SIZE)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if to_save:
        ensure_model_database()
        save_dir = generate_new_save()
        torch.save(model.state_dict(), get_model_filename(save_dir))
        config = {
            "VOCAB_SIZE": VOCAB_SIZE,
            "seed": SEED,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "context_length": seq_length,
            "num_epochs": num_epochs,
            "pattern_chance": pattern_chance,
        }
        with open(get_config_filename(save_dir), 'w') as f:
            json.dump(config, f, indent=4)

if __name__ == "__main__":
    model = AttentionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    train_model(model, optimizer, criterion, num_epochs=100000, miscellaneous_chance=True)

    sequence = generate_sequence(CONTEXT_LENGTH, repeated=REPEATED)
    text_sequence = " ".join(str(VOCAB_LIST[token]) for token in sequence)
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(sequence[:-1], dtype=torch.long).unsqueeze(0) # (1, seq_length - 1)
        outputs = model(inputs)["outputs"] # (1, VOCAB_SIZE)
        print()
        print("Generated sequence:", text_sequence)
        print()
        print("Next predicted token:", VOCAB_LIST[torch.argmax(outputs)])
        print()
