import random
import time
import torch
from requests import Response
import requests

# Prepare Prediction server config
prediction_server_url = "http://127.0.0.1:8000"
model_request = {
    "model_id": "multigram-predictor"
}

def request_prediction_progress(delay_secs = 5, timeout_secs = 600, log_prefix = "") -> Response:
    # keep requesting until condition met or times out
    for _ in range(timeout_secs // max(1, delay_secs)):
        if delay_secs > 0: # wait for progress to build up
            time.sleep(delay_secs)
        # check progress
        progress_resp = requests.get(f"{prediction_server_url}/progress/", params=model_request)
        progress_status, progress_body = progress_resp.status_code, progress_resp.json()
        print(f"{log_prefix}{progress_status=}")
        if progress_resp.status_code == 200:
            if len(progress_body["progress"]) > 0: # log info about progress
                costs = [progress["cost"] for progress in progress_body["progress"]]
                cost = sum(costs) / len(costs)
                avg_cost = progress_body["average_cost"]
                print(f"{cost=}")
                print(f"{avg_cost=}")
                last_epoch = progress_body["progress"][-1]["epoch"]
                print(f"{last_epoch=}")
            model_status = progress_body["status"]
            print(f"{model_status=}")
            if model_status == "Training":
                continue # checking
        else: # barf possible error body
            print(f"{progress_body=}")
            if progress_resp.status_code == 400: # malformed input
                continue # checking
        return progress_resp # done
    # timed out
    raise TimeoutError("Training took too long")

def make_prediction(input_vector: list[int]) -> list[float]:
    prediction_request = model_request | {
        "input": {
            "activation_vector": input_vector
        },
    }
    resp = requests.post(f"{prediction_server_url}/output/", json=prediction_request)

    if resp.status_code == 200:
        return resp.json()['output_vector']

    raise RuntimeError(f"Failed to receive a good prediction: {resp.status_code} - {resp.json()}")

def calculate_cost(cost_input_data: list[tuple]) -> float:
    cost_request = model_request | {
        "input": {
            "activation_vector": [input_data for input_data,_ in cost_input_data],
            "target_vector": [target for _,target in cost_input_data],
        },
    }
    resp = requests.post(f"{prediction_server_url}/output/", json=cost_request)

    if resp.status_code == 200:
        return resp.json()['cost']

    raise RuntimeError(f"Failed to calculate cost: {resp.status_code} - {resp.json()}")

def run_training(training_epochs: int, train_batch_size: int, train_data: list[tuple]):
    # Prepare training request parameters
    num_train_items = len(train_data)
    training_model_request = model_request | {
        "epochs": training_epochs,
        "learning_rate": 0.05,
        "batch_size": train_batch_size,
        "decay_rate": 0.9999847, # reduces learning rate by a factor of 10 every 150,000 epochs
        "dropout_rate": 0.0, # no drop out
        "l2_lambda": 0.0, # no L2 regularization
    }

    # Prepare training request
    training_request = training_model_request | {
        "training_data": [{
            "activation_vector": input_vector,
            "target_vector": target_vector,
        } for input_vector, target_vector in train_data],
    }
    print(f"Prepared training data of size {num_train_items} to run for {training_epochs} epochs "
          f"with batch size {train_batch_size}")

    timeout_secs = max(training_epochs // 10, 10)

    # Submit training request to prediction service
    training_resp = requests.put(f"{prediction_server_url}/train/", json=training_request)
    print(f"Submitted: {training_resp.status_code} - {training_resp.json()}")
    # check progress
    request_prediction_progress(timeout_secs=timeout_secs)
    # mark end of training request
    print(f"###### Finished Training Round ########")

if __name__ == "__main__":
    # User selection
    user_selection = input('Choose (S) generate samples or (T) perform training:').upper()
    print(f"{user_selection=}")

    # Read example in
    with open("example.txt", "r", encoding="utf-8") as f:
        example = f.read()
    example = example.lower() # work with lowercase only

    word_break = " "
    words = example.split(word_break)

    # Extract tokens
    tokens = sorted(list(set(ch for ch in example if ch.isalpha())))
    tokens.insert(0, word_break)
    num_tokens = len(tokens)
    print(f"{num_tokens=}")
    s2i = {s: i for i, s in enumerate(tokens)}
    print(f"{s2i=}")

    # declare block context size
    block_size = 8
    # embedding depth number of dimensions
    embed_depth = 24
    # declare block fusing size
    fuse_size = 2

    # Create prediction model if not already
    model_resp = request_prediction_progress(0, 1)
    if model_resp.status_code == 404:
        # Include embedding layer and hidden non-linear activation layer
        hidden_layer_size = 128
        create_model_request = model_request | {
            "layer_sizes": [
                # embedding layer
                num_tokens, embed_depth,
                # hidden layers
                embed_depth * fuse_size, hidden_layer_size,
                hidden_layer_size * fuse_size, hidden_layer_size,
                hidden_layer_size * fuse_size, hidden_layer_size,
                # output layer
                num_tokens,
            ],
            "weight_algo": "xavier",
            "bias_algo": "",
            "activation_algos": [
                # embedding layer
                "embedding",
                # hidden layers
                "flatten", "linear", "batchnorm", "tanh",
                "flatten", "linear", "batchnorm", "tanh",
                "flatten", "linear", "batchnorm", "tanh",
                # output layer
                "softmax",
            ],
            "optimizer": "stochastic",
            "batchnorm_eps": 1e-4,
            "batchnorm_momentum": 0.025,
            "confidence": 0.2,
        }
        create_model_resp = requests.post(f"{prediction_server_url}/model/", json=create_model_request)
        print(f"{create_model_resp.status_code} - {create_model_resp.json()}")
    elif model_resp.status_code != 200:
        raise RuntimeError(f"Prediction Service error: {model_resp.status_code} - {model_resp.json()}")

    # Perform according to user selection
    if user_selection == 'T':
        # Shuffle words
        random.shuffle(words)
        # Build training data
        training_data: list[tuple] = []
        # Start context with block of word breaks
        input_context = [0] * block_size
        for w in words:
            for s in w + word_break:
                if s in s2i.keys(): # check token is in vocabulary
                    label_idx = s2i[s]
                    # add to training data
                    training_data.append((input_context, [label_idx]))
                    # keep a running context of block size or reset context for word break
                    input_context = input_context[1:] + [label_idx] if label_idx > 0 else [0] * block_size

        # Preview training data
        for x, y in training_data[:21]:
            print(f"{''.join(tokens[ix] for ix in x)} --> {''.join(tokens[iy] for iy in y)}")

        # Build data splits
        num_training_items = len(training_data)
        print(f"{num_training_items=}")
        num_split_train_items = int(0.8 * num_training_items)
        print(f"{num_split_train_items=}")
        num_split_val_items = int(0.1 * num_training_items)
        print(f"{num_split_val_items=}")
        num_split_test_items = num_training_items - num_split_train_items - num_split_val_items
        print(f"{num_split_test_items=}")

        split_train_data = training_data[:num_split_train_items]
        split_val_data = training_data[num_split_train_items:num_split_train_items + num_split_val_items]
        split_test_data = training_data[num_split_train_items + num_split_val_items:]

        # Ask for training options
        num_training_epochs = int(input('How many epochs shall we perform training?'))
        print(f"{num_training_epochs=}")
        training_batch_size = int(input('Set training batch size='))
        print(f"{training_batch_size=}")

        # Run training on split
        run_training(num_training_epochs, training_batch_size, split_train_data)

        # Calculate cost on splits
        split_train_cost = calculate_cost(split_train_data)
        print(f"{split_train_cost=}")
        split_val_cost = calculate_cost(split_val_data)
        print(f"{split_val_cost=}")
        split_test_cost = calculate_cost(split_test_data)
        print(f"{split_test_cost=}")

    else: # Generate sample
        # Ask for number of words
        num_samples = int(input('How many samples would you like?'))
        print(f"{num_samples=}")

        # Build predictions
        token_indices: list[int] = [0] * block_size
        for sample_idx in range(num_samples * 2): # twice to skip any empty samples
            # Reset sample
            sample = ""
            # Generate tokens until word break seen for chosen number of words
            while len(sample) < 30: # avoid really long samples
                # Predict next token
                output_vector = make_prediction(token_indices)
                output_idx: int = torch.multinomial(torch.tensor(output_vector), num_samples=1).item()
                # Slide next token in for next prediction
                token_indices = token_indices[1:] + [output_idx]
                # Check word break
                if output_idx == 0:
                    break
                # Append next token
                sample += tokens[output_idx]
            # Present sample if not empty
            if sample:
                print(f"{sample=}")
            # Enough samples then break
            if sample_idx >= num_samples - 1:
                break
