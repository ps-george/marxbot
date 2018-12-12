# platobot
ChatBot resurrection of Plato

## AWS steps

1. Launch GPU EC2 instance from console
2. SSH/SCP to instance to copy cope across

```bash
# Make private key only readable by me
chmod 400 /path/my-key-pair.pem
# SSH into EC2
ssh -i /path/my-key-pair.pem ubuntu@ec2-198-51-100-1.compute-1.amazonaws.com
# Upload files to EC2 using SCP
scp -i /path/my-key-pair.pem /path/SampleFile.txt ubuntu@c2-198-51-100-1.compute-1.amazonaws.com:~
```


3. Edit code to save model weights to an S3 bucket (using boto3)
4. Can add functionality to load weights from S3 bucket also.


### To activate tensorflow on Deep Learning AMIs.
`source activate tensorflow_p36`


## TODO

- [x] Tidy up dataset
- [x] Store dataset somewhere
- [x] Write API
- [ ] Get report format and create report outline (names, structure, etc.)
- [x] Figure out plan for AWS/Google compute instances
- [x] Get network running on the cloud
- [ ] Use article to help create a Keras network that runs

## Notes

When training single RNN network, output goes bad between 20 and 30 epochs (overfitting).

