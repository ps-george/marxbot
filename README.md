# marxbot
ChatBot resurrection of Karl Marx

## Using the chatbot

1. Ensure data is available in directory 'marx.txt' and 'movie.txt'
2. Run person.py for a REPL loop with Marx.

## Loading weights

1. Install requirements `pip install -r requirements.txt`
2. Import MarxBot from model.py
3. Initialize the class MarxBot e.g. `marx = MarxBot(sources=['marx.txt'])`
4. Call `marx.load('path/to/weights')`

## Training

### Locally

1. Install requirements `pip install -r requirements.txt`
2. Import MarxBot from model.py
3. Initialize the class MarxBot e.g. `marx = MarxBot(sources=['marx.txt'])`
4. Call `marx.train()`

### On AWS

1. Launch GPU EC2 instance with Ubuntu Deep Learning AMI from AWS console
2. Ensure to give EC2 IAM write permissions to the s3 bucket to save weights and output to.
3. SSH/SCP to instance to copy code across

```bash
# Make private key only readable by me
chmod 400 /path/my-key-pair.pem
# SSH into EC2
ssh -i /path/my-key-pair.pem ubuntu@ec2-198-51-100-1.compute-1.amazonaws.com
# Upload files to EC2 using SCP
scp -i /path/my-key-pair.pem /path/SampleFile.txt ubuntu@c2-198-51-100-1.compute-1.amazonaws.com:~
```
4. Double check requirements are installed on the instance by running `pip install -r requirements.txt`
5. Initialize MarxBot with correct parameters `marx = MarxBot(s3bucket=BUCKETNAME)`
6. Call `marx.train_online()`, which will train the model and save progress to the s3bucket.


### To activate tensorflow on Deep Learning AMIs.
`source activate tensorflow_p36`

## Todo list
- [x] Tidy up dataset
- [x] Store dataset somewhere
- [x] Write API
- [x] Figure out plan for AWS/Google compute instances
- [x] Get network running on the cloud
- [x] Use article to help create a Keras network that runs

