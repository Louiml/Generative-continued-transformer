import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('idx_to_token.txt', 'r', encoding='utf-8') as f:
    idx_to_token = [line.strip() for line in f]

with open('token_to_idx.txt', 'r', encoding='utf-8') as f:
    token_to_idx = {line.strip().split()[0]: int(line.strip().split()[1]) for line in f}

class GCT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_text, hidden):
        embedded = self.embedding(input_text)
        output, hidden = self.gru(embedded, hidden)
        output = self.fc(output)
        return output, hidden

def generate_text(model, device, input_text, max_length=100):
    with torch.no_grad():
        input_tensor = torch.tensor(input_text, dtype=torch.long, device=device).unsqueeze(0)
        hidden = torch.zeros(1, 1, model.hidden_size, device=device)
        generated_text = []

        for _ in range(max_length):
            output, hidden = model(input_tensor, hidden)
            _, predicted = torch.max(output, 2)
            generated_token = idx_to_token[predicted.item()]

            if generated_token == '<END>':
                break

            if generated_token not in generated_text:
                generated_text.append(generated_token)

            input_tensor = predicted
            hidden = hidden.detach()

        return generated_text

input_text = input("Enter a few sentences as input: ")

tokenized_text = input_text.lower().split()

numerical_text = [token_to_idx.get(token, token_to_idx['<UNK>']) for token in tokenized_text]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCT(len(idx_to_token), 256, len(idx_to_token)).to(device)
torch.save(model.state_dict(), 'gct_model.pth')
model.load_state_dict(torch.load("gct_model.pth", map_location=device))
model.eval()

generated_text = generate_text(model, device, numerical_text)

print("Generated Text:")
print(' '.join(generated_text))
