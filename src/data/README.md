# Docstrings for data collection.

This source code to collect data from youtube video (url)

## Local machine
### Clone repository
```bash
cd dir/contain/project
git clone -b data_collection git@github.com:minhnv4099/vietnamese-av-asr.git
# Replace branch 'data_collection' by 'main' if available and necessary.
```
### Install dependencies
#### Test environment and update
```bash
which conda 
conda --version 
python --version 
conda install --channel defaults conda python=3.9 --yes
conda update --channel defaults --all --yes
```

#### Install hf-transfer
```bash
pip install hf-transfer
env HF_HUB_ENABLE_HF_TRANSFER=1
```

#### Install Coccoc tokenizer
```bash
pip install Cython
git clone https://github.com/coccoc/coccoc-tokenizer.git
cd coccoc-tokenizer
mkdir build
cd build
cmake -DBUILD_PYTHON=1 -DCMAKE_INSTALL_PREFIX=/usr/local ..
make install
cp /usr/local/lib/python3.9/site-packages/CocCocTokenizer-1.4-py3.9-linux-x86_64.egg/CocCocTokenizer.* /usr/local/lib/python3.9/site-packages
```
#### Test coccoc tokenizer
```bash
conda list | grep coccoctokenizer   # should show coccoctokenizer 1.4
```

#### Install requirements
```bash
cd dir/contain/project/vietnamese-av-asr/
pip install -r ./src/data/databases/requirements.txt
pip install -U datasets
pip install fsspec
```
### Login huggingface-hub
```bash
!huggingface-cli login
# Then add token to the prompt
```
### Process
Track URLs
```bash
python src/data/tasks/track.py --url <url> --channel-name <channel-name>
```
- `<url>: url of video in YouTube.`
- `<channel-name>: As alias channel to process.`

Put channel name to channel.txt file
```python
#channel name forms 'batch_00000'
channel = """
batch_99999
"""
with open('src/data/databases/channel.txt','w') as f:
    print(*channel, sep='\n',file=f)
```
Process data through each task
```bash
python src/data/tasks/process 
    --task <task> in [download|asd|crop|vndetect|transcribe] 
    --channel-names <channel-names> 
    --cache-dir <cache-dir>  
    --output-dir <output-dir> 
    --upload-to-hub          
    --clean-input            
    --clean-output           
```


- `<task>: Select one task to process.`
- `<channel-names>: Channel name or file contains channel names. Default 'src/data/databases/channel.txt'`
- `<cache-dir>: Directory contains downloaded data from hub. Default 'data/external/'`
- `<output-dir>: Directory contains processed data ready upload to hub. Default 'data/processed/'`
- `--upload-to-hub: Upload processed data to hub.`
- `--clean-input: Clean cache dir.`
- `--clean-output: Clean output dir.`

### Pipeline (demo)
Prepare url.
```python
# Paste urls
url = """
https://www.youtube.com/watch?v=xmBdARJx6PY
""".split()
with open("src/data/databases/url.txt", "w") as f:
    print(*url, sep="\n", file=f)
```
Execute tasks on video file or url.
```bash
python src/data/tasks/pipe.py
    --url <url>                     
    --file <path-to-file>
    --do-file                       
    --channel-name <channel-name>   
    --tasks <task1> <task2> ...     
    --cache-dir <cache-dir>         
    --output-dir <output-dir>       
    --clean-input
    --clean-output
```
- `<url>: Video url in YouTube. Default 'src/data/databases/url.txt'.`
- `<path-to-file>: Path to video file.`
- `--do-file: Process video file instead of url.`
- `<channel-name>: Name of channel contains url/file.`
- `<task1> <task2>: Select tasks to process, or 'full' to do all tasks.`
- `<cache-dir>: Directory contains downloaded data from hub. Default 'data/external/'`
- `<output-dir>: Directory contains processed data ready upload to hub. Default 'data/processed/'`
- `--clean-input: Clean cache dir.`
- `--clean-output: Clean output dir.`
- `Available tasks in order: track, download, asd, crop, vndetect, transcribe`
- `Notes: Should select tasks consecutively. `

## Colab
### Open browser, go to colab, Open notebook 'data_process.ipynb' in [repo](https://github.com/minhnv4099/vietnamese-av-asr/tree/data_collection/notebooks).
