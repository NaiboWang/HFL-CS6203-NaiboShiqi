# Federated Learning with secret sharing.

Implementation of the federated algorithm on `PyTorch` and conduct MPC with `MP-SPDZ` Secret sharing library.

Experiments are produced on 10 datasets such as MNIST, COVID19 and CIFAR10 (both IID and non-IID).

## **Environment Configuration**
Download docker file from:

https://drive.google.com/file/d/13ffiBG9eFv_vZGPtpN0gbbS_mwxjkiPI/view?usp=sharing

The docker file is round 19 GB, please download with patience. If you see a message `Sorry, you can't view or download this file at this time.`when downloading, you should first log in with your Google Account and then download, if you still cannot download after you download, you can contact me via email and I will copy the docker file to you. 

Please deploy our docker file on Linux system.

Since we provide a docker file for you to run, so basically there is no library you need to install manually. I.e., you don't need to configure the environment manually.

Note that you should run the following commands on **all the machines** you want to use to conduct the distributed experiments.

### **CPU Version Environment Configuration**

This part helps you to deploy our docker image on your machine to run our experiments on `CPU`, if you want to run them on `NVIDIA GPU`, you can follow the instructions in the next section.

Make sure you have installed `docker`, if not, you can install docker with the instructions from docker's official website: 

https://docs.docker.com/engine/install/

To load and run our docker container, please execute following commands:

0. Make sure you have installed `docker` and have `sudo` privilege.

1. Copy `hfl-naiboshiqi.tar` to your machine, such as put it in `/home/naibo/` and go to the directory.

2. Load docker image:

    ```shell
    sudo docker load < hfl-naiboshiqi.tar
    ```

3. Since the size of our docker file is around 23 GB, so the loading process may take about 4 to 5 minutes before you can see any output, please wait with patience. After loaded, you will see an image names `hfl-naiboshiqi` in the docker list image:

    ```shell
    sudo docker images
    ```

    and you will see one item like this:

    ```shell
    hfl-naiboshiqi   1.0   1c789cfa9c5a   2 hours ago   20.3GB
    ```

4. Make sure that the following ports are not occupied by other processes: 8080 to 8280, 14000 to 14200, 50000 to 50200, in total 600 ports. For example, you can see whether port `8080` is occupied by:

    ```shell
    lsof -i:8080
    ```

5. Then you can run our docker image:

    ```shell
    sudo docker run --ipc=host -p 8080-8280:8080-8280 -p 14000-14200:14000-14200 -p 50000-50200:50000-50200 -itd --name NaiboShiqi hfl-naiboshiqi:1.0  /bin/bash
    ```

6. After about 20 seconds, you will see a container with name `NaiboShiqi` running on your machine by:

    ```shell
    sudo docker ps
    ```

7. Our container will occupy 600 ports on your local machine to make sure this machine can support at most 200 clients as well as secret sharing backend. Then, you need to enter our container to run our commands by:
   
    ```shell
    sudo docker exec -it NaiboShiqi /bin/bash
    ```

After you entered our container, you need first compile the `MP-SPDZ` secret sharing library to make sure you can run the secret sharing backend on this machine. Of course, if are not going to run the secret sharing backend on this machine, you do not need to execute commands in the following steps. I.e., step `8 to 14` only needs to be run on the machine you want to run the `MP-SPDZ` secret sharing backend.
   
8. Go to the root path of MP-SPDZ folder:
   
    ```shell
    cd /root/HFL_CS6203_NaiboWang_ShiqiZhang/MP-SPDZ
    ```

9. Clean the compiled programs at first:

    ```shell
    make clean
    ```

10. Compile the virtual machine for semi-honest additive secret sharing protocols:

    ```shell
    make -j 8 semi2k-party.x
    ```

11. Compile the secret adding clients `secret-add.cpp`:

    ```shell
    make secret-add.x
    ```

12. Create triple shares for three parties (SPDZ engines):

    ```shell
    ./Scripts/setup-ssl.sh
    ```

13. Create SSL keys and certificates for `199` clients, which allows up to 199 clients:

    ```Shell
    ./Scripts/setup-clients.sh 199
    ```

14. Compile the high-level program in SPDZ engines:

    ```Shell
    ./compile.py add_by_socket
    ```


You can then run our experiments with the commands in the `Running the experiements` section.

Again, if you want to deploy our docker image on different machines, you need to run the above commands on every machine.

### **GPU Version Environment Configuration**

Note that above instructions are setting an environment for you to run experiments on CPU, if you want to run the experiments on your **NVIDIA GPU**, you can follow the following instructions.

First, run commands the same as step `1 to 4` at last section. 

Then, you need to install `nvidia drivers` and `nvidia-docker`, you can install `nvidia-docker` by following the installation guide from their official website:

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

And you can see your CUDA version by:

```shell
nvidia-smi
```

Make sure your CUDA with version at least `10.1`. After you installed `nvidia drivers` and `nvidia-docker`, you can then run our docker image by:

```shell
sudo docker run --gpus all --ipc=host -p 8080-8280:8080-8280 -p 14000-14200:14000-14200 -p 50000-50200:50000-50200 -itd --name NaiboShiqi hfl-naiboshiqi:1.0  /bin/bash
```

Then after you entered the container by:

```shell
sudo docker exec -it NaiboShiqi /bin/bash
```

you can run `nvidia-smi` to see if CUDA is supported by the container.

Then, you need to update PyTorch to match the CUDA version by following the instructions at the PyTorch official website (make sure you select PyTorch version as `1.7.0`):

https://pytorch.org/get-started/locally/

E.g., if your CUDA version is `11.0`, then you can update your PyTorch version to match CUDA 11.0:

```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
```

you can validate whether the PyTorch on the container support CUDA after you installed it by:

```shell
python
```

after you entered a python shell, you should run:

```python
import torch
torch.__version__
torch.cuda.is_available()
```

if you get `1.7.0` and `True`, that means you can use GPU to run PyTorch experiments on this container.

And also, if you want to run the `MP-SPDZ` secret sharing backend on this machine, you need to follow step `8 to 14` from the last section to compile the library.

Now you can run our experiments on your GPUs by setting the arguments `gpu=0` with the commands in the `Running the experiements` section.

## **Running the experiments**

You can use the `Command Helper` we designed to help you generate the command you want to run:

http://naibo.wang/Projects/CS6203/commandhelper.html

### **MP-SPDZ Secret sharing backend**

You need to start different MP-SPDZ secret sharing backend based on different scenarios.

* If you are running experiments on a single-machine simulation environment such as you are going to run `fedrated_main.py` which is the simulated version of federated averaging, you can start a session, enter the docker container and run the backend with 2 clients by:

    ```shell
    cd /root/HFL_CS6203_NaiboWang_ShiqiZhang/MP-SPDZ
    ./Scripts/cs6203.sh add_by_socket & python run_client.py 2
    ```

Note that every time you want to start a new experiment, you should **stop the past backend and run a new secret sharing backend**.

And note you need to **start another session** to run experiments and keep the current session (i.e., the secret sharing backend session) **alive**.

* If you are going to run the real distributed version of federated averaging algorithm with secret sharing, you should start specified command by the number of clients you are going to run. For example, if you want to run an experiment with in total `4` clients, you need to execute:
  
    ```shell
    cd /root/HFL_CS6203_NaiboWang_ShiqiZhang/MP-SPDZ
    ./Scripts/cs6203.sh add_by_socket & python run_client.py 4
    ```

    I.e., just need to change the arguments `<nclients>` clients which communicate with SPDZ engines:

    ```
    ./Scripts/cs6203.sh add_by_socket & python run_client.py <nclients>
    ```

Also, you need to **start another session** to run experiments and keep the current session (i.e., the secret sharing backend session) **alive**.



### **Federated Learning experiments**

If you want to run federated learning experiments with secret sharing, you should make sure you **started the MP-SPDZ secret sharing backend correctly**, as mentioned in the last section. 

First, you should know the DOCKER IP address `IP_DOCKER`, which is the ip address of the docker container, and the HOST IP address `IP_HOST` of the host machine which the docker container are running on.

Run `ifconfig` on your host machine **(NOT run inside docker container)** to get the HOST IP address, and run `ifconfig` inside docker to get the DOCKER IP address. Generally, the HOST IP address is like `172.26.186.87` and the DOCKER IP address is like `172.17.0.2`.

Suppose the DOCKER IP address of the secret sharing backend is `IP_DOCKER_A`, and the HOST IP address of the secret sharing backend is `IP_HOST_A`.



Before you run any experiment, you should start a new session and keep the secret sharing backend session alive, and change folder to the source code of Federated Learning Experiments:

```shell
sudo docker exec -it NaiboShiqi /bin/bash
cd /root/HFL_CS6203_NaiboWang_ShiqiZhang/Federated-Learnining-PyTorch/src
```

-----

#### **Baseline**
The baseline experiment trains the model in the conventional way.

* To run the baseline experiment with MNIST dataset on MLP using CPU:
    ```
    python baseline_main.py --dataset=mnist --epochs=10
    ```

* Or to run COVID19 dataset experiment on GPU (e.g.: if gpu:0 is available):
    ```
    python baseline_main.py --dataset=COVID19 --gpu=0 --epochs=10
    ```
-----
#### **Federated averaging in a single-machine simulation environment**.

Federated experiment involves training a global model using many local models.

* To run the federated experiment with CIFAR10 dataset on CPU (IID) `without` secret sharing, simulated 100 users with 10 users selected every epoch:

    ```
    python federated_main.py --dataset=cifar --iid=1 --secret_share=0 --epochs=100 --num_users=100 --frac=0.1
    ```

* To run the federated experiment with COVID19_twitter dataset on CPU `with` secret sharing under non-IID condition, simulated 100 users with 10 users selected every epoch.

    Make sure you have started the MP-SPDZ secret sharing backend with two clients, i.e., you have run the following command at another session first:

    ```shell
    ./Scripts/cs6203.sh add_by_socket & python run_client.py 2
    ```

    Then you can conduct the experiment:

    Do remember to replace `IP_HOST_A` in the end of the following command to the real **HOST IP address** of the secret sharing backend, if you are running experiment and secret sharing backend on the same machine with the same IP address, you can skip `--ss_address` or set `--ss_address=localhost`.

    ```shell
    python federated_main.py --dataset=COVID19_twitter --iid=0 --epochs=100 --secret_share=1 --num_users=100 --frac=0.1 --ss_address=IP_HOST_A
    ```
-----
#### **Federated averaging in real distributed environment**.

Suppose you want to run Federated averaging algorithm on heart_disease dataset with iid setting on 5 machines with 3 workers, 1 coordinator and 1 secret sharing backend, the details of the machines are shown below:

| ID | Role | HOST IP Address | DOCKER IP Address |
| :-----: | :----: | :----: | :----: |
| A | Secret-Share backend | 172.26.186.80 | 172.17.0.2 |
| B | Worker 1 | 172.26.186.81 | 172.17.0.2 |
| C | Worker 2 | 172.26.186.82 | 172.17.0.3 |
| D | Worker 3 | 172.26.186.83 | 172.17.0.4 |
| E | Coordinator | 172.26.186.84 | 172.17.0.2 |

To run federated averaging with secret sharing, you should load and run the docker container on all these five machines by the instructions at the `Environment Configuration` section.

Then, run the following commands:

1. First, we should first start the MP-SPDZ secret sharing backend at `Machine A`'s docker container, do compile the MP-SPDZ library first by the step 8 to 14 at the `Environment Configuration` section. Since we have 3 workers here, therefore the arguments should be set to 3:

    ```shell
    cd /root/HFL_CS6203_NaiboWang_ShiqiZhang/MP-SPDZ
    ./Scripts/cs6203.sh add_by_socket & python run_client.py 3
    ```

    If successfully started, the last line of the output will be:

    ```shell
    Party 0 received external client connection from client id: 2
    ```

2. Then we need to start the three workers in order, before that, we should split the dataset to three parts with non-iid setting for all workers. 
   
   On `Machine B`, you should **enter the docker container** and run the following commands:

    ```shell
    cd /root/HFL_CS6203_NaiboWang_ShiqiZhang/Federated-Learnining-PyTorch/src
    python dataset_split.py --dataset=heart_disease --num_users=3 --iid=1
    python fedavg_client.py --worker_number=0 --num_users=3 --secret_share=1 --dataset=heart_disease --ss_address=172.26.186.80
    ```

    Note that the argument `ss_address` whose value `172.26.186.80` is the **HOST** IP address of `Machine A`, which means the HOST IP address of the secret sharing backend, **please replace this address with your own backend HOST IP address when you are running your experiments**. And the argument `worker_number` is `0` which means the worker number for this machine is 0. Then you will see `worker 0 has started`, the worker will start to train after you have started the coordinator, now you just keep it here and wait.

    Then, you need to start worker 1 and 2 on `Machine C and D` by the almost same commands (just need to change the worker number).

    Run the following commands on `Machine C`:
    
    ```shell
    cd /root/HFL_CS6203_NaiboWang_ShiqiZhang/Federated-Learnining-PyTorch/src
    python dataset_split.py --dataset=heart_disease --num_users=3 --iid=1
    python fedavg_client.py --worker_number=1 --num_users=3 --secret_share=1 --dataset=heart_disease --ss_address=172.26.186.80
    ```

    Run the following commands on `Machine D`:

    ```shell
    cd /root/HFL_CS6203_NaiboWang_ShiqiZhang/Federated-Learnining-PyTorch/src
    python dataset_split.py --dataset=heart_disease --num_users=3 --iid=1
    python fedavg_client.py --worker_number=2 --num_users=3 --secret_share=1 --dataset=heart_disease --ss_address=172.26.186.80
    ```

    And you will see `Worker 1 started` and `Worker 2 started`.

3. Now the secret sharing backend and three workers are all started, we can now start the coordinator to start training. Run the following command on `Machine E` after you entered the docker container.
   
   First, configure the workers' addresses use vim:

   ```shell
   cd /root/HFL_CS6203_NaiboWang_ShiqiZhang/Federated-Learnining-PyTorch/src
   vim ../configuration.json
   ```

   For this case, you should modify the `configuration.json` as follows:

   ```json
    {
        "worker_list":[
            "http://172.26.186.81:50001",
            "http://172.26.186.82:50002",
            "http://172.26.186.83:50003"
        ]
    }
   ```
   
   `http` is the communication protocol we are using, `172.26.186.81` to `172.26.186.83` are the **HOST IP addresses** of Worker 0 to 2. `50001` to `50003` are the ports three workers binding. If you want to add more workers, just add items follow `http://HOST_IP:PORT`, for which `PORT` is 50001+worker_number. 
   
   And don't set the `HOST_IP` as `localhost` or `127.0.0.1` even if the worker (client) and the coordinator are in the same machine, in this situation please set the `HOST_IP` as the `DOCKER IP Address` such as `172.17.0.2` (Again, please use `ifconfig` in the docker container to check for the `DOCKER IP Address`). 
   
   Save the configuration file by `:wq` at vim.

    And then, we also need to split the dataset first by:

    ```shell
    python dataset_split.py --dataset=heart_disease --num_users=3 --iid=1
    ```

    Then, you can start the coordinator and wait for the results:

    ```shell
    python fedavg_coordinator.py --epochs=100 --num_users=3 --frac=0.7 --secret_share=1 --dataset=heart_disease
    ```

    For this case, every round the coordinator will select 3x0.7, i.e., 2 workers to train.

If you want to run the distributed federated averaging algorithm without secret sharing, just set `--secret_share=0` for all three workers and the coordinator. After the training, you can quit all workers by `Ctrl+C` one by one.

-----
#### **Federated brain and federated brain_v2 algorithm in a single-machine simulation environment**

Distributed train models without a central server, averaging local models with new version. You can see details about federated_brain and federated_brain_v2 algorithm from the report.

* To run the federated brain experiment with femnist dataset on CPU (IID) `without` secret sharing, simulated 100 users with 10 users selected every epoch:

    ```shell
    python federated_brain.py --dataset=femnist --iid=1 --secret_share=0 --epochs=100 --num_users=100 --frac=0.1
    ```

* To run the federated brain_v2 experiment with heartbeat dataset on CPU `with` secret sharing under IID condition, simulated 100 users with 10 users selected every epoch.

    Make sure you have started the MP-SPDZ secret sharing backend with two clients, i.e., you have run the following command at another session first:

    ```shell
    ./Scripts/cs6203.sh add_by_socket & python run_client.py 2
    ```

    Then you can conduct the experiment:

    ```shell
    python federated_brain_v2.py --dataset=heartbeat --iid=1 --epochs=100 --secret_share=1 --num_users=100 --frac=0.1 --ss_address=IP_HOST_A
    ```

    Do remember to replace `IP_HOST_A` to the real **HOST IP address** of the secret sharing backend, if you are running experiment and secret sharing backend on the same machine with the same IP address, you can skip `--ss_address` or set `--ss_address=localhost`.

-----
#### **Federated distributed random algorithm in a single-machine simulation environment**

Random select args.frac * args.num_users clients to train based on their local models in order, the i th model is initialized from the i-1 th model.

* To run the federated distributed random experiment with malaria_cell_images dataset on CPU (IID) `without` secret sharing, simulated 100 users with 10 users selected every epoch:

    ```
    python federated_distributed_random.py --dataset=malaria_cell_images --iid=1 --secret_share=0 --epochs=100 --num_users=100 --frac=0.1
    ```

* To run the federated distributed random experiment with chest_xray dataset on CPU `with` secret sharing under non-IID condition, simulated 100 users with 10 users selected every epoch.

    Make sure you have started the MP-SPDZ secret sharing backend with two clients, i.e., you have run the following command at another session first:

    ```shell
    ./Scripts/cs6203.sh add_by_socket & python run_client.py 2
    ```

    Then you can conduct the experiment:

    ```shell
    python federated_distributed_random.py --dataset=chest_xray --iid=0 --epochs=100 --secret_share=1 --num_users=100 --frac=0.1 --ss_address=IP_HOST_A
    ```

    Do remember to replace `IP_HOST_A` to the real **HOST IP address** of the secret sharing backend, if you are running experiment and secret sharing backend on the same machine with the same IP address, you can skip `--ss_address` or set `--ss_address=localhost`.

----
Among all the experiments, you can change the default values of other parameters to simulate different conditions. Refer to the `options` section.

## **Options**
The default values for various parameters parsed to the experiment are given in ```options.py```. Details are given for those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'fmnist', 'cifar', 'COVID19', 'chest_xray', 'heart_disease', 'heartbeat', 'COVID19_twitter', 'malaria_cell_images', 'HAM10000'.
* ```--secret_share:```    Default: 0. Set `--secret_share=1` to run experiments with secret sharing.
* ```--optimizer:``` Default: 'adam'. Options: 'sgd', 'adam'.
* ```--gpu:```      Default: -1 (runs on CPU). Can also be set to the specific gpu id (if you have only 1 gpu, just set `--gpu=0`).
* ```--parallel:``` you can set it to 1 if you have multiple GPUs on your machine.
* ```--epochs:```   Number of rounds of training, default: 100.
* ```--lr:```       Learning rate, set to 0.001 by default.
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:```Number of users. Default is 3.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.7.
* ```--local_ep:``` Number of local training epochs in each user. Default is 5.
* ```--local_bs:``` Batch size of local updates in each user. Default is 50.
* ```--worker_number:``` The worker number of a client.
* ```--address:``` The IP address of a client, default: 172.17.0.2.
* ```--ss_address:``` The IP address of the address of secret sharing backend, default: 172.17.0.2.