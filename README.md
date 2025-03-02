# Federated Learning Contribution Estimation (FLCE)

Welcome to the Federated Learning Contribution Estimation (FLCE) repository!

Federated Learning (FL) is a collaborative machine learning paradigm where multiple clients utilize their data for model training without transferring their data. 
In this context, Federated Learning Contribution Estimation (FLCE) emerges, aiming to
compute fair and reasonable contribution scores as incentives to motivate FL clients.
This project is dedicated to providing methods and tools for estimating the contributions of clients in an FL environment.

The project stems from a paper to appear at VLDB'24 entitled "Contributions Estimation in Federated Learning: A Comprehensive Experimental Evaluation".

## Features

- **FLCE Methods**: Implements a variety of FLCE methods to estimate the contribution of each client.
- **Benchmarking Tools**: Includes tools for comparing methods in terms of effectiveness, robustness, and efficiency.

[//]: # (- **Simulation Environment**: Provides a simulated federated learning environment for testing and benchmarking contribution estimation methods under various conditions.)

[//]: # (- **Extensible Framework**: Designed to be easily extended with new contribution estimation algorithms and federated learning models. You may add new datasets, data distributions, models, and flce methods to the project.)

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements.

Create and a conda environment and activate it using
```shell
conda env create -f environment.yml
conda activate flce
```

Install pytorch using
```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

[//]: # (### Installation)

[//]: # ()
[//]: # (To perform FLCE, follow these steps:)

[//]: # ()
[//]: # (1. Clone the repository:)

[//]: # (   ```)

[//]: # (   git clone https://github.com/veevang/flce)

[//]: # (   ```)

[//]: # (2. Navigate to the repository directory:)

[//]: # (   ```)

[//]: # (   cd flce)

[//]: # (   ```)

[//]: # (3. Install the required Python packages:????)

[//]: # (   ```)

[//]: # (   pip install -r requirements.txt)

[//]: # (   ```)

### Usage

[//]: # (To start using FLCE, you can run the main script with the following command:)

[//]: # (```shell)

[//]: # (python -u main.py --num_parts 8 -t effective -m $method --dataset tictactoe --model TicTacToeMLP --lr 0.005 --num_epoch 60 --hidden_layer_size 16 --batch_size 16 --device cpu -a 0.65 --distribution "quantity skew" -s $seed --num_repeat 1 --start_date $today --num_try $sub_folder --value_functions "${value_functions[@]}" > "./log/tictactoe quantity skew $method.out" 2>&1 &)

[//]: # (```)

[//]: # (conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge)

[//]: # (-c conda-forge scikit-learn)

For examples, please refer to the `run.sh` file in the root directory.
After editing the file, you might use the following code to run the experiments:
```shell
bash run.sh
```

[//]: # (## Contributing)

[//]: # ()
[//]: # (We welcome contributions from the community! If you have improvements or new features you'd like to add, please follow these steps:)

[//]: # ()
[//]: # (1. Fork the repository.)

[//]: # (2. Create a new branch for your feature &#40;`git checkout -b feature/AmazingFeature`&#41;.)

[//]: # (3. Commit your changes &#40;`git commit -m 'Add some AmazingFeature'`&#41;.)

[//]: # (4. Push to the branch &#40;`git push origin feature/AmazingFeature`&#41;.)

[//]: # (5. Open a Pull Request.)

[//]: # ()
[//]: # (For more detailed instructions, see [CONTRIBUTING.md]&#40;./CONTRIBUTING.md&#41;.)

[//]: # (## License)

[//]: # ()
[//]: # (This project is licensed under the [INSERT LICENSE HERE] License - see the [LICENSE]&#40;LICENSE&#41; file for details.)

## Acknowledgments

- Thank Prof. Guoliang Li, Dr. Yong Wang, and Dr. Kaiyu Li for their contributions to the development of this project.

[//]: # (- In our implementation, we referred to the repository: https://github.com/AI-secure/Shapley-Study.???)

[//]: # (- Thank funding???)

## Contact

For any questions or feedback regarding FLCE, please contact us at [chen-yw20@mails.tsinghua.edu.cn](mailto:chen-yw20@mails.tsinghua.edu.cn) or post an issue on github.

---

In summary, this project can help evaluate FLCE methods. We hope this project can help you understand existing FLCE methods and design your own ones. Enjoy!
