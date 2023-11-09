def plot_training(artefact_folder_: str, title_: str):
    """
    Plots curves for training and evaluation loss (Y axes 1) and accuracy (Y axes 2) against epoch
    - training: blue curve (0 - max)
    - evaluation loss: orange curve (0 - max)
    - accuracy: green curve (0-1)

    :param artefact_folder_: "./models/fromScratch_WBC_1/"
    :param title_: "Training Plots for WBC1, from scratch"
    :return: na
    """

    # Read log_history.json file
    log_filename = "trainer_state.json"
    log_filepath = os.path.join(artefact_folder_, log_filename)

    with open(log_filepath, "r") as f:
        json_data = json.load(f)
        log_data = json_data.get("log_history", None)
        # log_data = json_data["log_history"]

    # Initialize lists to hold tuple pairs (value, epoch)
    training_loss = []
    eval_loss = []
    eval_accuracy = []

    # Extract relevant data
    for entry in log_data:
        epoch = entry.get('epoch', None)
        if epoch is not None:
            if 'loss' in entry:
                training_loss.append((entry['loss'], epoch))
            if 'eval_loss' in entry:
                eval_loss.append((entry['eval_loss'], epoch))
            if 'eval_accuracy' in entry:
                eval_accuracy.append((entry['eval_accuracy'], epoch))

    # Sort tuple pairs based on epoch
    training_loss.sort(key=lambda x: x[1])
    eval_loss.sort(key=lambda x: x[1])
    eval_accuracy.sort(key=lambda x: x[1])

    # Create the plot
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot([epoch for val, epoch in training_loss], [val for val, epoch in training_loss], label='Training Loss',
             color='tab:blue')
    ax1.plot([epoch for val, epoch in eval_loss], [val for val, epoch in eval_loss], label='Evaluation Loss',
             color='tab:orange')
    ax1.tick_params(axis='y')
    ax1.set_ylim(bottom=0.0)  # Set range for loss to start from 0.0
    # ax1.grid(linestyle='none')  # Remove y-axis gridlines for ax1

    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Accuracy', color='tab:green')
    # ax2.plot([epoch for val, epoch in eval_accuracy], [val for val, epoch in eval_accuracy],
    #          label='Evaluation Accuracy', color='tab:green')
    # ax2.tick_params(axis='y')
    # ax2.set_ylim(0, 1.0)  # Set range for accuracy to be 0-1.0
    # # ax2.grid(linestyle='none')  # Remove y-axis gridlines for ax1

    # Add title and legend
    plt.title(title_)
    fig.tight_layout()
    fig.legend(loc="center right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    # fig.legend(loc="center right")

    save_path = os.path.join(artefact_folder_, 'training_curve.png')
    plt.savefig(save_path)
    plt.show()
