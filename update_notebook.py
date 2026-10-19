import json

notebook_path = 'assignment_1_pytorch_mnist_skeleton_2026.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 5: Net architecture to expose net inputs, activations after ReLU, and Max-Pooling
cell5_code = """# Defining Architecture,loss and optimizer
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 10)

        # Dictionary to store intermediate net inputs and activations
        self.intermediates = {}

    def forward(self, x):
        # Layer 1: conv1
        conv1_net = self.conv1(x)               # net input
        conv1_relu = F.relu(conv1_net)          # activation after ReLU
        conv1_pool = F.max_pool2d(conv1_relu, 2)# activation after Max-Pooling

        # Layer 2: conv2
        conv2_net = self.conv2(conv1_pool)      # net input
        conv2_relu = F.relu(conv2_net)          # activation after ReLU
        conv2_pool = F.max_pool2d(conv2_relu, 2)# activation after Max-Pooling

        # Flatten from 64*4*4 to 1024
        flat = conv2_pool.view(-1, 1024)

        # Layer 3: fc1
        fc1_net = self.fc1(flat)                # net input
        fc1_relu = F.relu(fc1_net)              # activation after ReLU
        fc1_drop = self.drop(fc1_relu)

        # Layer 4: fc2
        fc2_net = self.fc2(fc1_drop)            # net input
        out = F.softmax(fc2_net, dim=1)         # final output

        # Cache intermediate terms for TensorBoard monitoring
        self.intermediates = {
            'conv1': {
                'net_input': conv1_net,
                'relu': conv1_relu,
                'maxpool': conv1_pool
            },
            'conv2': {
                'net_input': conv2_net,
                'relu': conv2_relu,
                'maxpool': conv2_pool
            },
            'fc1': {
                'net_input': fc1_net,
                'relu': fc1_relu
            },
            'fc2': {
                'net_input': fc2_net
            }
        }

        return out
"""
nb['cells'][5]['source'] = [line + '\n' for line in cell5_code.splitlines()]

# Update Cell 7: Net instantiation + stats logger helper function
cell7_code = """model = Net()
optimizer = optim.Adam(model.parameters(), lr=lr)
writer = SummaryWriter(log_dir=logging_dir)
# Defining the test and trainig loops
eps = 1e-13

def log_tensor_statistics(writer, tag_prefix, tensor, step):
    \"\"\"
    Helper function to log min, max, mean, std, and histogram to TensorBoard.
    \"\"\"
    t = tensor.detach().float()
    writer.add_scalar(f'{tag_prefix}/min', t.min().item(), step)
    writer.add_scalar(f'{tag_prefix}/max', t.max().item(), step)
    writer.add_scalar(f'{tag_prefix}/mean', t.mean().item(), step)
    writer.add_scalar(f'{tag_prefix}/std', t.std().item(), step)
    writer.add_histogram(f'{tag_prefix}/histogram', t.cpu().numpy(), step)
"""
nb['cells'][7]['source'] = [line + '\n' for line in cell7_code.splitlines()]

# Update Cell 11: Training loop
cell11_code = """def train(epoch):
    model.train()

    criterion = nn.NLLLoss(reduction='none')

    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(torch.log(output + eps), target) # = sum_k(-t_k * log(y_k))
        loss = loss.mean() # average over the batch
        loss.backward()
        optimizer.step()

        # Step index for global training steps
        step = (epoch - 1) * len(train_loader) + batch_idx

        # Monitor training loss during training (Part a)
        writer.add_scalar('train/loss_step', loss.item(), step)

        if batch_idx % logging_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\\tLoss: {loss.item():.6f}')

    # Monitor training loss per epoch
    writer.add_scalar('train/loss_epoch', loss.item(), epoch)

    # Monitor statistics (min, max, mean, standard deviation, histogram) after each epoch (Part b)
    # 1. Weights and biases
    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:] # 'weight' or 'bias'
        log_tensor_statistics(writer, f'parameters/{layer}/{attr}', param, epoch)

    # 2. Net inputs, activations after ReLU, and activations after Max-Pooling at each layer
    for layer_name, layer_dict in model.intermediates.items():
        for term_name, tensor_val in layer_dict.items():
            log_tensor_statistics(writer, f'activations/{layer_name}/{term_name}', tensor_val, epoch)
"""
nb['cells'][11]['source'] = [line + '\n' for line in cell11_code.splitlines()]

# Update Cell 13: Testing loop
cell13_code = """def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0

    criterion = nn.NLLLoss(reduction='none')

    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)

            test_loss += criterion(torch.log(output + eps), target).sum().item() # sum up batch loss
            pred = output.argmax(dim=1) # get the index of the max log-probability
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    test_error = 100.0 - test_accuracy
    print(f'Test Epoch: {epoch}\\tLoss: {test_loss:.6f}\\tAccuracy: {test_accuracy:.2f}%\\tTest Error: {test_error:.2f}%')

    # Monitor test metrics after each epoch
    writer.add_scalar('test/loss', test_loss, epoch)
    writer.add_scalar('test/accuracy', test_accuracy, epoch)
    writer.add_scalar('test/error', test_error, epoch)
"""
nb['cells'][13]['source'] = [line + '\n' for line in cell13_code.splitlines()]

# Add TensorBoard display cell at the end if not present
has_tb = any('%tensorboard' in ''.join(c.get('source', [])) for c in nb['cells'])
if not has_tb:
    nb['cells'].append({
        'cell_type': 'markdown',
        'id': 'tensorboard_markdown',
        'metadata': {},
        'source': [
            "## Launch TensorBoard\\n",
            "Run the cell below to inspect training loss, test error, and statistics (min, max, mean, std, histogram) for weights, biases, and layer activations.\\n"
        ]
    })
    nb['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'id': 'tensorboard_code',
        'metadata': {},
        'outputs': [],
        'source': [
            "# Load the TensorBoard notebook extension\\n",
            "%load_ext tensorboard\\n",
            "# Launch TensorBoard inside the notebook (points to log directory or default 'runs' directory)\\n",
            "%tensorboard --logdir runs\\n"
        ]
    })

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Updated notebook successfully!')

