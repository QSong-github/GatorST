# main.py - Main training/testing entry point
# util.py - Utility functions: seed setting, metrics, mixup, format_time
# data_loader.py - Data loading and preprocessing logic, CellDataset class
# model.py - Model definition, loss functions, train and test functions


if __name__ == '__main__':
    import os
    import json
    import time
    from data_loader import loader_construction
    from model import train, test
    from util import setup_seed

    data_folder = "./data"
    save_results_path = "./result.json"
    save_model_folder = "./saved_models"

    os.makedirs(save_model_folder, exist_ok=True)
    all_results = {}

    for file_name in os.listdir(data_folder):
        if file_name.endswith(".h5ad"):
            data_name = os.path.splitext(file_name)[0]
            if data_name not in ['15_processed_all']:  # Add your target datasets
                continue
            if data_name in all_results or 'Xenium' in data_name:
                continue

            print(f'Start Running {data_name}')
            epochs = 50
            batch_size = 20
            data_path = os.path.join(data_folder, file_name)

            train_loader, val_loader, test_loader, input_dim, n_clusters, n_clusters_test = loader_construction(
                data_name, data_path, batch_size
            )

            all_results[data_name] = []
            for run in range(10):
                seed = run
                setup_seed(seed)

                save_model_path = os.path.join(save_model_folder, f"{data_name}_model_run_{run}")

                best_epoch, min_loss = train(
                    train_loader, val_loader, lr=0.001, seed=seed, epochs=epochs,
                    n_clusters=n_clusters, input_dim=input_dim, save_model_path=save_model_path
                )

                results = test(
                    test_loader, n_clusters, n_clusters_test, input_dim,
                    save_model_path, seed
                )

                all_results[data_name].append(results)

            with open(save_results_path, "w") as json_file:
                json.dump(all_results, json_file, indent=4)

            print(f"Results saved to {save_results_path}")
